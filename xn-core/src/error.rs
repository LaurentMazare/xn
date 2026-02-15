use crate::{DType, Shape};

impl std::fmt::Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

/// Main library error type.
#[derive(thiserror::Error)]
pub enum Error {
    // === DType Errors ===
    #[error("{msg}, expected: {expected:?}, got: {got:?}")]
    UnexpectedDType { msg: &'static str, expected: DType, got: DType },

    #[error("dtype mismatch in {op}, lhs: {lhs:?}, rhs: {rhs:?}")]
    DTypeMismatchBinaryOp { lhs: DType, rhs: DType, op: &'static str },

    // === Shape Errors ===
    #[error("{msg}, expected: {expected:?}, got: {got:?}")]
    UnexpectedShape { msg: String, expected: Shape, got: Shape },

    #[error("shape mismatch in {op}, lhs: {lhs:?}, rhs: {rhs:?}")]
    ShapeMismatchBinaryOp { lhs: Shape, rhs: Shape, op: &'static str },

    #[error("unexpected number of dims, expected {expected}, got shape {shape:?}")]
    UnexpectedNumberOfDims { expected: usize, shape: Shape },

    #[error("dim out of range, shape: {shape:?}, dim: {dim}, op: {op}")]
    DimOutOfRange { shape: Shape, dim: i64, op: &'static str },

    #[error("duplicate dim index, shape: {shape:?}, dims: {dims:?}, op: {op}")]
    DuplicateDimIndex { shape: Shape, dims: Vec<usize>, op: &'static str },

    #[error("matmul shape mismatch: {msg}, lhs: {lhs:?}, rhs: {rhs:?}")]
    MatmulShapeMismatch { msg: &'static str, lhs: Shape, rhs: Shape },

    /// Utf8 parse error.
    #[error(transparent)]
    FromUtf8(#[from] std::string::FromUtf8Error),

    /// I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// SafeTensor error.
    #[error(transparent)]
    SafeTensor(#[from] safetensors::SafeTensorError),

    /// Arbitrary errors wrapping.
    #[error("{0}")]
    Wrapped(Box<dyn std::fmt::Display + Send + Sync>),

    #[error("{context}\n{inner}")]
    Context { inner: Box<Self>, context: Box<dyn std::fmt::Display + Send + Sync> },

    /// Adding path information to an error.
    #[error("path: {path:?} {inner}")]
    WithPath { inner: Box<Self>, path: std::path::PathBuf },

    #[error("{inner}\n{backtrace}")]
    WithBacktrace { inner: Box<Self>, backtrace: Box<std::backtrace::Backtrace> },

    /// User generated error message, typically created via `bail!`.
    #[error("{0}")]
    Msg(String),

    #[error("unwrap none")]
    UnwrapNone,

    #[cfg(feature = "cuda")]
    #[error(transparent)]
    Cublas(#[from] cudarc::cublas::result::CublasError),

    #[cfg(feature = "cuda")]
    #[error(transparent)]
    CudaDriver(#[from] cudarc::driver::DriverError),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn wrap(err: impl std::fmt::Display + Send + Sync + 'static) -> Self {
        Self::Wrapped(Box::new(err)).bt()
    }

    pub fn msg(err: impl std::fmt::Display) -> Self {
        Self::Msg(err.to_string()).bt()
    }

    pub fn debug(err: impl std::fmt::Debug) -> Self {
        Self::Msg(format!("{err:?}")).bt()
    }

    pub fn bt(self) -> Self {
        let backtrace = std::backtrace::Backtrace::capture();
        match backtrace.status() {
            std::backtrace::BacktraceStatus::Disabled
            | std::backtrace::BacktraceStatus::Unsupported => self,
            _ => Self::WithBacktrace { inner: Box::new(self), backtrace: Box::new(backtrace) },
        }
    }

    pub fn with_path<P: AsRef<std::path::Path>>(self, p: P) -> Self {
        Self::WithPath { inner: Box::new(self), path: p.as_ref().to_path_buf() }
    }

    pub fn context(self, c: impl std::fmt::Display + Send + Sync + 'static) -> Self {
        Self::Context { inner: Box::new(self), context: Box::new(c) }
    }
}

#[macro_export]
macro_rules! bail {
    ($msg:literal $(,)?) => {
        return Err($crate::Error::Msg(format!($msg).into()).bt())
    };
    ($err:expr $(,)?) => {
        return Err($crate::Error::Msg(format!($err).into()).bt())
    };
    ($fmt:expr, $($arg:tt)*) => {
        return Err($crate::Error::Msg(format!($fmt, $($arg)*).into()).bt())
    };
}

pub fn zip<T, U>(r1: Result<T>, r2: Result<U>) -> Result<(T, U)> {
    match (r1, r2) {
        (Ok(r1), Ok(r2)) => Ok((r1, r2)),
        (Err(e), _) => Err(e),
        (_, Err(e)) => Err(e),
    }
}

// Taken from anyhow.
pub trait Context<T> {
    /// Wrap the error value with additional context.
    fn context<C>(self, context: C) -> Result<T>
    where
        C: std::fmt::Display + Send + Sync + 'static;

    /// Wrap the error value with additional context that is evaluated lazily
    /// only once an error does occur.
    fn with_context<C, F>(self, f: F) -> Result<T>
    where
        C: std::fmt::Display + Send + Sync + 'static,
        F: FnOnce() -> C;
}

impl<T> Context<T> for Option<T> {
    fn context<C>(self, context: C) -> Result<T>
    where
        C: std::fmt::Display + Send + Sync + 'static,
    {
        match self {
            Some(v) => Ok(v),
            None => Err(Error::UnwrapNone.context(context).bt()),
        }
    }

    fn with_context<C, F>(self, f: F) -> Result<T>
    where
        C: std::fmt::Display + Send + Sync + 'static,
        F: FnOnce() -> C,
    {
        match self {
            Some(v) => Ok(v),
            None => Err(Error::UnwrapNone.context(f()).bt()),
        }
    }
}

pub(crate) fn check_same_shape(lhs: &Shape, rhs: &Shape, op: &'static str) -> Result<()> {
    if lhs != rhs {
        Err(Error::ShapeMismatchBinaryOp { lhs: lhs.clone(), rhs: rhs.clone(), op }.bt())
    } else {
        Ok(())
    }
}
