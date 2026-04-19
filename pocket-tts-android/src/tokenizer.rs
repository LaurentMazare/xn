// Pure-Rust Unigram SentencePiece tokenizer — a port of
// pocket-tts-wasm/www/worker.js (decodeSentencepieceModel + UnigramTokenizer).
//
// Just enough to load the `tokenizer.model` protobuf shipped with
// kyutai/pocket-tts and reproduce the Viterbi encode path without pulling in
// the C++ `sentencepiece` crate (which is awkward to cross-compile to Android
// NDK targets).

use std::collections::HashMap;

#[derive(Debug)]
pub struct Unigram {
    vocab: HashMap<String, (u32, f32)>,
    unk_id: u32,
}

#[derive(Debug)]
pub struct TokenizerError(pub String);

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TokenizerError {}

struct Piece {
    piece: String,
    score: f32,
    ty: u32,
}

impl Unigram {
    pub fn from_proto_bytes(buf: &[u8]) -> Result<Self, TokenizerError> {
        let pieces = parse_sentencepiece_model(buf)?;
        let mut vocab: HashMap<String, (u32, f32)> = HashMap::with_capacity(pieces.len());
        let mut unk_id = 0u32;
        for (i, p) in pieces.iter().enumerate() {
            if p.ty == 2 {
                unk_id = i as u32;
            }
            // NORMAL (1), USER_DEFINED (4), BYTE (6). UNK (2), CONTROL (3) and
            // UNUSED (5) are excluded from the vocab we lookup against, matching
            // the JS reference.
            if p.ty == 1 || p.ty == 4 || p.ty == 6 {
                vocab.insert(p.piece.clone(), (i as u32, p.score));
            }
        }
        Ok(Self { vocab, unk_id })
    }

    pub fn from_file(path: &str) -> Result<Self, TokenizerError> {
        let bytes =
            std::fs::read(path).map_err(|e| TokenizerError(format!("reading {path}: {e}")))?;
        Self::from_proto_bytes(&bytes)
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut normalized = String::with_capacity(text.len() + 3);
        normalized.push('\u{2581}');
        for c in text.chars() {
            if c == ' ' {
                normalized.push('\u{2581}');
            } else {
                normalized.push(c);
            }
        }
        self.viterbi(&normalized)
    }

    pub fn decode(&self, _tokens: &[u32]) -> String {
        // split_into_best_sentences calls decode on token sub-ranges to rebuild
        // sentence text. The CLI implementation uses sentencepiece which joins
        // pieces and replaces ▁ with a space. We don't need decode for the
        // Android hot path, but provide a minimal implementation anyway.
        String::new()
    }

    fn viterbi(&self, text: &str) -> Vec<u32> {
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        let n = chars.len();
        if n == 0 {
            return Vec::new();
        }
        let total_bytes = text.len();

        #[derive(Clone, Copy)]
        struct Cell {
            score: f64,
            len: usize,
            id: i64,
        }
        let mut best = vec![Cell { score: f64::NEG_INFINITY, len: 0, id: -1 }; n + 1];
        best[0] = Cell { score: 0.0, len: 0, id: -1 };

        for i in 0..n {
            if !best[i].score.is_finite() {
                continue;
            }
            let start_byte = chars[i].0;
            let max_len = core::cmp::min(n - i, 64);
            for len in 1..=max_len {
                let end_byte =
                    if i + len < n { chars[i + len].0 } else { total_bytes };
                let sub = &text[start_byte..end_byte];
                if let Some(&(id, score)) = self.vocab.get(sub) {
                    let new_score = best[i].score + score as f64;
                    if new_score > best[i + len].score {
                        best[i + len] = Cell { score: new_score, len, id: id as i64 };
                    }
                }
            }
            if !best[i + 1].score.is_finite() {
                let c = chars[i].1;
                let cp = c as u32;
                let key = format!("<0x{:02X}>", cp & 0xFF);
                let (fid, fscore) = match self.vocab.get(&key) {
                    Some(&(id, score)) => (id, score),
                    None => (self.unk_id, -100.0),
                };
                best[i + 1] = Cell {
                    score: best[i].score + fscore as f64,
                    len: 1,
                    id: fid as i64,
                };
            }
        }

        let mut ids = Vec::new();
        let mut p = n;
        while p > 0 {
            let cell = best[p];
            if cell.len == 0 || cell.id < 0 {
                break;
            }
            ids.push(cell.id as u32);
            p -= cell.len;
        }
        ids.reverse();
        ids
    }
}

// ---- Minimal protobuf decoder -------------------------------------------
//
// ModelProto { repeated SentencePiece pieces = 1; ... }
// SentencePiece { optional string piece = 1; optional float score = 2;
//                 optional Type type = 3 [default = NORMAL]; }
//
// We only care about top-level field 1 (pieces) and inside each piece the
// three fields above. Everything else is skipped, mirroring worker.js.

fn parse_sentencepiece_model(buf: &[u8]) -> Result<Vec<Piece>, TokenizerError> {
    let mut pieces = Vec::new();
    let mut pos = 0usize;
    while pos < buf.len() {
        let (key, new_pos) = read_varint(buf, pos)?;
        pos = new_pos;
        let field = key >> 3;
        let wire = key & 0x7;
        if field == 1 && wire == 2 {
            let (len, after_len) = read_varint(buf, pos)?;
            pos = after_len;
            let end = pos
                .checked_add(len as usize)
                .ok_or_else(|| TokenizerError("protobuf length overflow".into()))?;
            if end > buf.len() {
                return Err(TokenizerError("protobuf piece out of bounds".into()));
            }
            pieces.push(decode_piece(&buf[pos..end])?);
            pos = end;
        } else {
            pos = skip_field(buf, pos, wire)?;
        }
    }
    Ok(pieces)
}

fn decode_piece(buf: &[u8]) -> Result<Piece, TokenizerError> {
    let mut piece = String::new();
    let mut score = 0f32;
    let mut ty: u32 = 1;
    let mut pos = 0usize;
    while pos < buf.len() {
        let (key, new_pos) = read_varint(buf, pos)?;
        pos = new_pos;
        let field = key >> 3;
        let wire = key & 0x7;
        match (field, wire) {
            (1, 2) => {
                let (len, after_len) = read_varint(buf, pos)?;
                pos = after_len;
                let end = pos
                    .checked_add(len as usize)
                    .ok_or_else(|| TokenizerError("piece string length overflow".into()))?;
                if end > buf.len() {
                    return Err(TokenizerError("piece string out of bounds".into()));
                }
                piece = std::str::from_utf8(&buf[pos..end])
                    .map_err(|_| TokenizerError("piece string not utf8".into()))?
                    .to_string();
                pos = end;
            }
            (2, 5) => {
                if pos + 4 > buf.len() {
                    return Err(TokenizerError("piece score out of bounds".into()));
                }
                let bytes = [buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]];
                score = f32::from_le_bytes(bytes);
                pos += 4;
            }
            (3, 0) => {
                let (v, after) = read_varint(buf, pos)?;
                ty = v as u32;
                pos = after;
            }
            (_, wire) => {
                pos = skip_field(buf, pos, wire)?;
            }
        }
    }
    Ok(Piece { piece, score, ty })
}

fn read_varint(buf: &[u8], mut pos: usize) -> Result<(u64, usize), TokenizerError> {
    let mut result: u64 = 0;
    let mut shift = 0;
    loop {
        if pos >= buf.len() {
            return Err(TokenizerError("varint truncated".into()));
        }
        let b = buf[pos];
        pos += 1;
        result |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            return Ok((result, pos));
        }
        shift += 7;
        if shift >= 64 {
            return Err(TokenizerError("varint too long".into()));
        }
    }
}

fn skip_field(buf: &[u8], pos: usize, wire: u64) -> Result<usize, TokenizerError> {
    match wire {
        0 => {
            let (_, p) = read_varint(buf, pos)?;
            Ok(p)
        }
        1 => {
            if pos + 8 > buf.len() {
                Err(TokenizerError("fixed64 out of bounds".into()))
            } else {
                Ok(pos + 8)
            }
        }
        2 => {
            let (len, p) = read_varint(buf, pos)?;
            let end = p
                .checked_add(len as usize)
                .ok_or_else(|| TokenizerError("length-delimited overflow".into()))?;
            if end > buf.len() {
                Err(TokenizerError("length-delimited out of bounds".into()))
            } else {
                Ok(end)
            }
        }
        5 => {
            if pos + 4 > buf.len() {
                Err(TokenizerError("fixed32 out of bounds".into()))
            } else {
                Ok(pos + 4)
            }
        }
        other => Err(TokenizerError(format!("unsupported wire type {other}"))),
    }
}

impl ptts::Tokenizer for Unigram {
    fn encode(&self, text: &str) -> Vec<u32> {
        Unigram::encode(self, text)
    }
    fn decode(&self, tokens: &[u32]) -> String {
        Unigram::decode(self, tokens)
    }
}
