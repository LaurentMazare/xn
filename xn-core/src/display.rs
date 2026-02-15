//! Pretty printing of tensors
//!
//! This implementation should be in line with the [PyTorch version](https://github.com/pytorch/pytorch/blob/7b419e8513a024e172eae767e24ec1b849976b13/torch/_tensor_str.py).
//!
use crate::{Backend, Result, Tensor, WithDType, WithDTypeF};

/// Options for Tensor pretty printing
#[derive(Debug, Clone)]
pub struct PrinterOptions {
    pub precision: usize,
    pub threshold: usize,
    pub edge_items: usize,
    pub line_width: usize,
    pub sci_mode: Option<bool>,
}

static PRINT_OPTS: std::sync::Mutex<PrinterOptions> =
    std::sync::Mutex::new(PrinterOptions::const_default());

impl PrinterOptions {
    // We cannot use the default trait as it's not const.
    const fn const_default() -> Self {
        Self { precision: 4, threshold: 1000, edge_items: 3, line_width: 80, sci_mode: None }
    }
}

pub fn print_options() -> &'static std::sync::Mutex<PrinterOptions> {
    &PRINT_OPTS
}

pub fn set_print_options(options: PrinterOptions) {
    *PRINT_OPTS.lock().unwrap() = options
}

pub fn set_print_options_default() {
    *PRINT_OPTS.lock().unwrap() = PrinterOptions::const_default()
}

pub fn set_print_options_short() {
    *PRINT_OPTS.lock().unwrap() = PrinterOptions {
        precision: 2,
        threshold: 1000,
        edge_items: 2,
        line_width: 80,
        sci_mode: None,
    }
}

pub fn set_print_options_full() {
    *PRINT_OPTS.lock().unwrap() = PrinterOptions {
        precision: 4,
        threshold: usize::MAX,
        edge_items: 3,
        line_width: 80,
        sci_mode: None,
    }
}

pub fn set_line_width(line_width: usize) {
    PRINT_OPTS.lock().unwrap().line_width = line_width
}

pub fn set_precision(precision: usize) {
    PRINT_OPTS.lock().unwrap().precision = precision
}

pub fn set_edge_items(edge_items: usize) {
    PRINT_OPTS.lock().unwrap().edge_items = edge_items
}

pub fn set_threshold(threshold: usize) {
    PRINT_OPTS.lock().unwrap().threshold = threshold
}

pub fn set_sci_mode(sci_mode: Option<bool>) {
    PRINT_OPTS.lock().unwrap().sci_mode = sci_mode
}

struct FmtSize {
    current_size: usize,
}

impl FmtSize {
    fn new() -> Self {
        Self { current_size: 0 }
    }

    fn final_size(self) -> usize {
        self.current_size
    }
}

impl std::fmt::Write for FmtSize {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.current_size += s.len();
        Ok(())
    }
}

pub trait TensorFormatter: Sized {
    type Elem: WithDType;

    fn new<B: Backend>(t: &Tensor<Self::Elem, B>, po: &PrinterOptions) -> Result<Self>;

    fn fmt<W: std::fmt::Write>(&self, v: Self::Elem, max_w: usize, f: &mut W) -> std::fmt::Result;

    fn max_width<B: Backend>(&self, to_display: &Tensor<Self::Elem, B>) -> usize {
        let mut max_width = 1;
        if let Ok(vs) = to_display.flatten_all().and_then(|t| t.to_vec()) {
            for v in vs.iter().copied() {
                let mut fmt_size = FmtSize::new();
                let _res = self.fmt(v, 1, &mut fmt_size);
                max_width = usize::max(max_width, fmt_size.final_size())
            }
        }
        max_width
    }

    fn write_newline_indent(i: usize, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f)?;
        for _ in 0..i {
            write!(f, " ")?
        }
        Ok(())
    }

    fn fmt_tensor<B: Backend>(
        &self,
        t: &Tensor<Self::Elem, B>,
        indent: usize,
        max_w: usize,
        summarize: bool,
        po: &PrinterOptions,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        let dims = t.dims();
        let edge_items = po.edge_items;
        write!(f, "[")?;
        match dims {
            [] => {
                if let Ok(vs) = t.to_vec()
                    && let Some(&v) = vs.first()
                {
                    self.fmt(v, max_w, f)?
                }
            }
            [v] if summarize && *v > 2 * edge_items => {
                if let Ok(slice) = t.narrow(0, ..edge_items).and_then(|t| t.contiguous())
                    && let Ok(vs) = slice.to_vec()
                {
                    for v in vs.into_iter() {
                        self.fmt(v, max_w, f)?;
                        write!(f, ", ")?;
                    }
                }
                write!(f, "...")?;
                if let Ok(slice) = t.narrow(0, v - edge_items..*v).and_then(|t| t.contiguous())
                    && let Ok(vs) = slice.to_vec()
                {
                    for v in vs.into_iter() {
                        write!(f, ", ")?;
                        self.fmt(v, max_w, f)?;
                    }
                }
            }
            [_] => {
                let elements_per_line = usize::max(1, po.line_width / (max_w + 2));
                if let Ok(vs) = t.to_vec() {
                    for (i, v) in vs.into_iter().enumerate() {
                        if i > 0 {
                            if i % elements_per_line == 0 {
                                write!(f, ",")?;
                                Self::write_newline_indent(indent, f)?
                            } else {
                                write!(f, ", ")?;
                            }
                        }
                        self.fmt(v, max_w, f)?
                    }
                }
            }
            _ => {
                if summarize && dims[0] > 2 * edge_items {
                    for i in 0..edge_items {
                        match get_subtensor(t, i) {
                            Ok(sub) => {
                                self.fmt_tensor(&sub, indent + 1, max_w, summarize, po, f)?
                            }
                            Err(e) => write!(f, "{e:?}")?,
                        }
                        write!(f, ",")?;
                        Self::write_newline_indent(indent, f)?
                    }
                    write!(f, "...")?;
                    Self::write_newline_indent(indent, f)?;
                    for i in dims[0] - edge_items..dims[0] {
                        match get_subtensor(t, i) {
                            Ok(sub) => {
                                self.fmt_tensor(&sub, indent + 1, max_w, summarize, po, f)?
                            }
                            Err(e) => write!(f, "{e:?}")?,
                        }
                        if i + 1 != dims[0] {
                            write!(f, ",")?;
                            Self::write_newline_indent(indent, f)?
                        }
                    }
                } else {
                    for i in 0..dims[0] {
                        match get_subtensor(t, i) {
                            Ok(sub) => {
                                self.fmt_tensor(&sub, indent + 1, max_w, summarize, po, f)?
                            }
                            Err(e) => write!(f, "{e:?}")?,
                        }
                        if i + 1 != dims[0] {
                            write!(f, ",")?;
                            Self::write_newline_indent(indent, f)?
                        }
                    }
                }
            }
        }
        write!(f, "]")?;
        Ok(())
    }
}

/// Get subtensor at index i along dimension 0.
fn get_subtensor<T: WithDType, B: Backend>(t: &Tensor<T, B>, i: usize) -> Result<Tensor<T, B>> {
    let slice = t.narrow(0, i..i + 1)?.contiguous()?;
    // Remove the first dimension
    let new_dims: Vec<usize> = t.dims()[1..].to_vec();
    if new_dims.is_empty() { slice.reshape(vec![1]) } else { slice.reshape(new_dims) }
}

pub struct FloatFormatter<S: WithDType> {
    int_mode: bool,
    sci_mode: bool,
    precision: usize,
    _phantom: std::marker::PhantomData<S>,
}

impl<S> TensorFormatter for FloatFormatter<S>
where
    S: WithDTypeF,
{
    type Elem = S;

    fn new<B: Backend>(t: &Tensor<S, B>, po: &PrinterOptions) -> Result<Self> {
        let mut int_mode = true;
        let mut sci_mode = false;

        // Rather than containing all values, this should only include
        // values that end up being displayed according to [threshold].
        let values = t
            .flatten_all()?
            .to_vec()?
            .into_iter()
            .filter(|v: &S| v.is_finite() && !v.is_zero())
            .collect::<Vec<_>>();
        if !values.is_empty() {
            let mut nonzero_finite_min = S::max_value();
            let mut nonzero_finite_max = S::min_value();
            for &v in values.iter() {
                let v = v.abs();
                if v < nonzero_finite_min {
                    nonzero_finite_min = v
                }
                if v > nonzero_finite_max {
                    nonzero_finite_max = v
                }
            }

            for &value in values.iter() {
                if value.ceil() != value {
                    int_mode = false;
                    break;
                }
            }
            if let Some(v1) = S::from(1000.)
                && let Some(v2) = S::from(1e8)
                && let Some(v3) = S::from(1e-4)
            {
                sci_mode = nonzero_finite_max / nonzero_finite_min > v1
                    || nonzero_finite_max > v2
                    || nonzero_finite_min < v3
            }
        }

        match po.sci_mode {
            None => {}
            Some(v) => sci_mode = v,
        }
        Ok(Self { int_mode, sci_mode, precision: po.precision, _phantom: std::marker::PhantomData })
    }
    fn fmt<W: std::fmt::Write>(&self, v: Self::Elem, max_w: usize, f: &mut W) -> std::fmt::Result {
        if self.sci_mode {
            write!(f, "{v:width$.prec$e}", v = v, width = max_w, prec = self.precision)
        } else if self.int_mode {
            if v.is_finite() {
                write!(f, "{v:width$.0}.", v = v, width = max_w - 1)
            } else {
                write!(f, "{v:max_w$.0}")
            }
        } else {
            write!(f, "{v:width$.prec$}", v = v, width = max_w, prec = self.precision)
        }
    }
}

pub struct IntFormatter<S: WithDType> {
    _phantom: std::marker::PhantomData<S>,
}

impl<S> TensorFormatter for IntFormatter<S>
where
    S: WithDType + std::fmt::Display,
{
    type Elem = S;

    fn new<B: Backend>(_t: &Tensor<S, B>, _po: &PrinterOptions) -> Result<Self> {
        Ok(Self { _phantom: std::marker::PhantomData })
    }
    fn fmt<W: std::fmt::Write>(&self, v: Self::Elem, max_w: usize, f: &mut W) -> std::fmt::Result {
        write!(f, "{v:max_w$}")
    }
}

fn get_summarized_data<T: WithDType, B: Backend>(
    t: &Tensor<T, B>,
    edge_items: usize,
) -> Result<Tensor<T, B>> {
    let dims = t.dims();
    if dims.is_empty() {
        t.copy()
    } else if dims.len() == 1 {
        if dims[0] > 2 * edge_items {
            Tensor::cat(
                &[
                    &t.narrow(0, ..edge_items)?.contiguous()?,
                    &t.narrow(0, dims[0] - edge_items..)?.contiguous()?,
                ],
                0,
            )
        } else {
            t.copy()
        }
    } else if dims[0] > 2 * edge_items {
        let mut vs: Vec<Tensor<T, B>> = (0..edge_items)
            .map(|i| {
                let sub = get_subtensor(t, i)?;
                get_summarized_data(&sub, edge_items)
            })
            .collect::<Result<Vec<_>>>()?;
        for i in (dims[0] - edge_items)..dims[0] {
            let sub = get_subtensor(t, i)?;
            vs.push(get_summarized_data(&sub, edge_items)?)
        }
        let vs_refs: Vec<&Tensor<T, B>> = vs.iter().collect();
        Tensor::cat(&vs_refs, 0)
    } else {
        let vs: Vec<Tensor<T, B>> = (0..dims[0])
            .map(|i| {
                let sub = get_subtensor(t, i)?;
                get_summarized_data(&sub, edge_items)
            })
            .collect::<Result<Vec<_>>>()?;
        let vs_refs: Vec<&Tensor<T, B>> = vs.iter().collect();
        Tensor::cat(&vs_refs, 0)
    }
}

// Debug implementation for all tensor types
impl<T: WithDType, B: Backend> std::fmt::Debug for Tensor<T, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Tensor[")?;
        match self.dims() {
            [] => {
                if let Ok(vs) = self.to_vec()
                    && let Some(v) = vs.first()
                {
                    write!(f, "{v}")?
                }
            }
            [s] if *s < 10 => {
                if let Ok(vs) = self.to_vec() {
                    for (i, v) in vs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{v}")?;
                    }
                }
            }
            dims => {
                write!(f, "dims ")?;
                for (i, d) in dims.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{d}")?;
                }
            }
        }
        write!(f, "; {:?}]", self.dtype())
    }
}

// Display implementation for float types
impl<T: WithDType, B: Backend> std::fmt::Display for Tensor<T, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let po = PRINT_OPTS.lock().unwrap();
        let summarize = self.elem_count() > po.threshold;
        let to_display = if summarize {
            match get_summarized_data(self, po.edge_items) {
                Ok(v) => v,
                Err(err) => return write!(f, "{err:?}"),
            }
        } else {
            match self.copy() {
                Ok(v) => v,
                Err(err) => return write!(f, "{err:?}"),
            }
        };
        if let Ok(tf) = T::Formatter::new(&to_display, &po) {
            let max_w = tf.max_width(&to_display);
            tf.fmt_tensor(self, 1, max_w, summarize, &po, f)?;
            writeln!(f)?;
        }
        write!(f, "Tensor[{:?}, {:?}]", self.dims(), self.dtype())
    }
}
