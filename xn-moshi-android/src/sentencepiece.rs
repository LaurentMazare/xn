// Pure-Rust SentencePiece piece decoder. ASR only emits token ids; we just
// need id -> piece string lookup, so we skip the C++ `sentencepiece` crate
// (awkward to cross-compile to Android NDK targets) and parse the protobuf
// model file directly.
//
// ModelProto {
//   message SentencePiece {
//     optional string piece = 1;
//     optional float  score = 2;
//     optional Type   type  = 3;  // NORMAL=1 UNK=2 CONTROL=3 USER_DEFINED=4
//                                 // UNUSED=5 BYTE=6
//   }
//   repeated SentencePiece pieces = 1;
//   ...
// }

#[derive(Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

#[derive(Clone)]
struct Piece {
    piece: String,
    ty: u32,
}

pub struct SpDecoder {
    pieces: Vec<Piece>,
}

impl SpDecoder {
    pub fn from_proto_bytes(buf: &[u8]) -> Result<Self, Error> {
        let pieces = parse_model_proto(buf)?;
        Ok(Self { pieces })
    }

    pub fn from_file(path: &str) -> Result<Self, Error> {
        let bytes = std::fs::read(path).map_err(|e| Error(format!("reading {path}: {e}")))?;
        Self::from_proto_bytes(&bytes)
    }

    /// Decode a sequence of piece ids back to UTF-8 text, replicating the
    /// SentencePiece convention: pieces concatenate, the meta-space U+2581
    /// becomes a regular space, BYTE-type pieces decode to their raw byte,
    /// and CONTROL/UNK/UNUSED pieces are dropped.
    pub fn decode_piece_ids(&self, ids: &[u32]) -> String {
        // First pass: flatten ids to a Vec<u8> of utf8 (or replacement bytes
        // for BYTE pieces). Doing the byte stitching here means a sequence of
        // BYTE pieces that together form a multi-byte UTF-8 codepoint will be
        // reassembled correctly.
        let mut bytes: Vec<u8> = Vec::with_capacity(ids.len() * 2);
        for &id in ids {
            let Some(p) = self.pieces.get(id as usize) else {
                continue;
            };
            match p.ty {
                // NORMAL / USER_DEFINED
                1 | 4 => bytes.extend_from_slice(p.piece.as_bytes()),
                // BYTE: piece is "<0xNN>"; emit the actual byte
                6 => {
                    if let Some(b) = parse_byte_piece(&p.piece) {
                        bytes.push(b);
                    }
                }
                // UNK (2), CONTROL (3), UNUSED (5): skip
                _ => {}
            }
        }
        let s = String::from_utf8_lossy(&bytes).into_owned();
        // U+2581 (▁) marks word boundaries. We turn them into spaces; the
        // caller decides whether to strip the leading one (usually only for
        // the very first emission of an utterance).
        s.replace('\u{2581}', " ")
    }
}

fn parse_byte_piece(s: &str) -> Option<u8> {
    let s = s.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(s, 16).ok()
}

fn parse_model_proto(buf: &[u8]) -> Result<Vec<Piece>, Error> {
    let mut pieces = Vec::new();
    let mut pos = 0usize;
    while pos < buf.len() {
        let (key, np) = read_varint(buf, pos)?;
        pos = np;
        let field = key >> 3;
        let wire = key & 0x7;
        if field == 1 && wire == 2 {
            let (len, np) = read_varint(buf, pos)?;
            pos = np;
            let end = pos
                .checked_add(len as usize)
                .ok_or_else(|| Error("piece length overflow".into()))?;
            if end > buf.len() {
                return Err(Error("piece out of bounds".into()));
            }
            pieces.push(decode_piece(&buf[pos..end])?);
            pos = end;
        } else {
            pos = skip_field(buf, pos, wire)?;
        }
    }
    Ok(pieces)
}

fn decode_piece(buf: &[u8]) -> Result<Piece, Error> {
    let mut piece = String::new();
    let mut ty: u32 = 1;
    let mut pos = 0usize;
    while pos < buf.len() {
        let (key, np) = read_varint(buf, pos)?;
        pos = np;
        let field = key >> 3;
        let wire = key & 0x7;
        match (field, wire) {
            (1, 2) => {
                let (len, np) = read_varint(buf, pos)?;
                pos = np;
                let end = pos
                    .checked_add(len as usize)
                    .ok_or_else(|| Error("piece string overflow".into()))?;
                if end > buf.len() {
                    return Err(Error("piece string out of bounds".into()));
                }
                piece = std::str::from_utf8(&buf[pos..end])
                    .map_err(|_| Error("piece not utf8".into()))?
                    .to_string();
                pos = end;
            }
            (2, 5) => {
                if pos + 4 > buf.len() {
                    return Err(Error("score out of bounds".into()));
                }
                pos += 4;
            }
            (3, 0) => {
                let (v, np) = read_varint(buf, pos)?;
                ty = v as u32;
                pos = np;
            }
            (_, wire) => pos = skip_field(buf, pos, wire)?,
        }
    }
    Ok(Piece { piece, ty })
}

fn read_varint(buf: &[u8], mut pos: usize) -> Result<(u64, usize), Error> {
    let mut result: u64 = 0;
    let mut shift = 0;
    loop {
        if pos >= buf.len() {
            return Err(Error("varint truncated".into()));
        }
        let b = buf[pos];
        pos += 1;
        result |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            return Ok((result, pos));
        }
        shift += 7;
        if shift >= 64 {
            return Err(Error("varint too long".into()));
        }
    }
}

fn skip_field(buf: &[u8], pos: usize, wire: u64) -> Result<usize, Error> {
    match wire {
        0 => Ok(read_varint(buf, pos)?.1),
        1 => {
            if pos + 8 > buf.len() {
                Err(Error("fixed64 oob".into()))
            } else {
                Ok(pos + 8)
            }
        }
        2 => {
            let (len, p) = read_varint(buf, pos)?;
            let end = p.checked_add(len as usize).ok_or_else(|| Error("len overflow".into()))?;
            if end > buf.len() { Err(Error("len-delim oob".into())) } else { Ok(end) }
        }
        5 => {
            if pos + 4 > buf.len() {
                Err(Error("fixed32 oob".into()))
            } else {
                Ok(pos + 4)
            }
        }
        other => Err(Error(format!("unsupported wire type {other}"))),
    }
}
