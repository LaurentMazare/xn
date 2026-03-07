use crate::conv::{
    PadMode, StreamingConv1d, StreamingConv1dState, StreamingConvTr1dState,
    StreamingConvTranspose1d,
};
use xn::nn::var_builder::Path;
use xn::{Backend, Result, Tensor, WithDTypeF};

pub struct ConvDownsample1d<T: WithDTypeF, B: Backend> {
    conv: StreamingConv1d<T, B>,
}

impl<T: WithDTypeF, B: Backend> ConvDownsample1d<T, B> {
    pub fn load(vb: &Path<B>, stride: usize, dimension: usize) -> Result<Self> {
        let conv = StreamingConv1d::load(
            &vb.pp("conv"),
            dimension,
            dimension,
            2 * stride,
            stride,
            1,
            PadMode::Replicate,
            1,
            false,
        )?;
        Ok(Self { conv })
    }

    pub fn init_state(&self, batch_size: usize) -> Result<StreamingConv1dState<T, B>> {
        self.conv.init_state(batch_size)
    }

    pub fn forward(
        &self,
        x: &Tensor<T, B>,
        state: &mut StreamingConv1dState<T, B>,
    ) -> Result<Tensor<T, B>> {
        self.conv.forward(x, state)
    }

    /// Non-streaming forward (creates and discards state).
    pub fn forward_no_state(&self, x: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        let b = x.dim(0usize)?;
        let mut state = self.init_state(b)?;
        self.conv.forward(x, &mut state)
    }
}

pub struct ConvTrUpsample1d<T: WithDTypeF, B: Backend> {
    convtr: StreamingConvTranspose1d<T, B>,
}

impl<T: WithDTypeF, B: Backend> ConvTrUpsample1d<T, B> {
    pub fn load(vb: &Path<B>, stride: usize, dimension: usize) -> Result<Self> {
        let convtr = StreamingConvTranspose1d::load(
            &vb.pp("convtr"),
            dimension,
            dimension,
            2 * stride,
            stride,
            dimension, // groups = dimension (depthwise)
            false,
        )?;
        Ok(Self { convtr })
    }

    pub fn init_state(&self, batch_size: usize) -> Result<StreamingConvTr1dState<T, B>> {
        self.convtr.init_state(batch_size)
    }

    pub fn forward(
        &self,
        x: &Tensor<T, B>,
        state: &mut StreamingConvTr1dState<T, B>,
    ) -> Result<Tensor<T, B>> {
        self.convtr.forward(x, state)
    }
}
