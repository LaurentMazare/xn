use xn::{Backend, Result, Tensor, TensorView};

/// Macro to generate tests for both CPU and CUDA backends.
/// Each test function takes a device reference and runs the test logic.
macro_rules! test_both_backends {
    ($test_name:ident, $test_fn:ident) => {
        paste::paste! {
            #[test]
            fn [<$test_name _cpu>]() -> Result<()> {
                $test_fn(&xn::CPU)
            }

            #[cfg(feature = "cuda")]
            #[test]
            fn [<$test_name _cuda>]() -> Result<()> {
                let device = xn::cuda_backend::Device::new(0)?;
                $test_fn(&device)
            }
        }
    };
}

// =============================================================================
// Cat tests
// =============================================================================

fn test_cat_dim0_impl<B: Backend>(dev: &B) -> Result<()> {
    // Two 2x3 tensors concatenated along dim 0 -> 4x3
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), dev)?;
    let b: Tensor<f32, B> = Tensor::from_vec(vec![7., 8., 9., 10., 11., 12.], (2, 3), dev)?;

    let c = Tensor::cat(&[&a, &b], 0)?;
    assert_eq!(c.dims(), &[4, 3]);
    assert_eq!(c.to_vec()?, vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]);
    Ok(())
}
test_both_backends!(test_cat_dim0, test_cat_dim0_impl);

fn test_cat_dim1_impl<B: Backend>(dev: &B) -> Result<()> {
    // Two 2x3 tensors concatenated along dim 1 -> 2x6
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), dev)?;
    let b: Tensor<f32, B> = Tensor::from_vec(vec![7., 8., 9., 10., 11., 12.], (2, 3), dev)?;

    let c = Tensor::cat(&[&a, &b], 1)?;
    assert_eq!(c.dims(), &[2, 6]);
    // Row 0: [1,2,3] ++ [7,8,9] = [1,2,3,7,8,9]
    // Row 1: [4,5,6] ++ [10,11,12] = [4,5,6,10,11,12]
    assert_eq!(c.to_vec()?, vec![1., 2., 3., 7., 8., 9., 4., 5., 6., 10., 11., 12.]);
    Ok(())
}
test_both_backends!(test_cat_dim1, test_cat_dim1_impl);

fn test_cat_3d_dim1_impl<B: Backend>(dev: &B) -> Result<()> {
    // Two 2x2x3 tensors concatenated along dim 1 -> 2x4x3
    let a: Tensor<f32, B> = Tensor::from_vec((1..=12).map(|x| x as f32).collect(), (2, 2, 3), dev)?;
    let b: Tensor<f32, B> =
        Tensor::from_vec((13..=24).map(|x| x as f32).collect(), (2, 2, 3), dev)?;

    let c = Tensor::cat(&[&a, &b], 1)?;
    assert_eq!(c.dims(), &[2, 4, 3]);
    // Batch 0: [[1,2,3],[4,5,6]] ++ [[13,14,15],[16,17,18]]
    // Batch 1: [[7,8,9],[10,11,12]] ++ [[19,20,21],[22,23,24]]
    assert_eq!(
        c.to_vec()?,
        vec![
            1., 2., 3., 4., 5., 6., 13., 14., 15., 16., 17., 18., // batch 0
            7., 8., 9., 10., 11., 12., 19., 20., 21., 22., 23., 24. // batch 1
        ]
    );
    Ok(())
}
test_both_backends!(test_cat_3d_dim1, test_cat_3d_dim1_impl);

// =============================================================================
// Reshape tests
// =============================================================================

fn test_reshape_impl<B: Backend>(dev: &B) -> Result<()> {
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), dev)?;

    // Reshape to 3x2
    let b = a.reshape((3, 2))?;
    assert_eq!(b.dims(), &[3, 2]);
    assert_eq!(b.to_vec()?, vec![1., 2., 3., 4., 5., 6.]);

    // Reshape to 6
    let c = a.reshape((6,))?;
    assert_eq!(c.dims(), &[6]);

    // Reshape to 1x6
    let d = a.reshape((1, 6))?;
    assert_eq!(d.dims(), &[1, 6]);

    // Reshape to 1x2x3
    let e = a.reshape((1, 2, 3))?;
    assert_eq!(e.dims(), &[1, 2, 3]);
    Ok(())
}
test_both_backends!(test_reshape, test_reshape_impl);

fn test_reshape_with_hole_impl<B: Backend>(dev: &B) -> Result<()> {
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), dev)?;

    // Reshape with inferred dimension
    let b = a.reshape((3, ()))?;
    assert_eq!(b.dims(), &[3, 2]);

    let c = a.reshape(((), 2))?;
    assert_eq!(c.dims(), &[3, 2]);

    let d = a.reshape((1, (), 3))?;
    assert_eq!(d.dims(), &[1, 2, 3]);
    Ok(())
}
test_both_backends!(test_reshape_with_hole, test_reshape_with_hole_impl);

// =============================================================================
// Index select tests
// =============================================================================

fn test_index_select_dim0_impl<B: Backend>(dev: &B) -> Result<()> {
    // Select rows from a 4x3 tensor
    let a: Tensor<f32, B> =
        Tensor::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.], (4, 3), dev)?;

    // Select rows 0, 2, 3
    let b = a.index_select(&[0, 2, 3], 0)?;
    assert_eq!(b.dims(), &[3, 3]);
    assert_eq!(b.to_vec()?, vec![1., 2., 3., 7., 8., 9., 10., 11., 12.]);

    // Select with repetition
    let c = a.index_select(&[1, 1, 0], 0)?;
    assert_eq!(c.dims(), &[3, 3]);
    assert_eq!(c.to_vec()?, vec![4., 5., 6., 4., 5., 6., 1., 2., 3.]);
    Ok(())
}
test_both_backends!(test_index_select_dim0, test_index_select_dim0_impl);

fn test_index_select_dim1_impl<B: Backend>(dev: &B) -> Result<()> {
    // Select columns from a 2x4 tensor
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8.], (2, 4), dev)?;

    // Select columns 0, 2
    let b = a.index_select(&[0, 2], 1)?;
    assert_eq!(b.dims(), &[2, 2]);
    // Row 0: [1, 3], Row 1: [5, 7]
    assert_eq!(b.to_vec()?, vec![1., 3., 5., 7.]);
    Ok(())
}
test_both_backends!(test_index_select_dim1, test_index_select_dim1_impl);

fn test_index_select_3d_impl<B: Backend>(dev: &B) -> Result<()> {
    // 2x3x2 tensor
    let a: Tensor<f32, B> = Tensor::from_vec((1..=12).map(|x| x as f32).collect(), (2, 3, 2), dev)?;
    // Data layout:
    // Batch 0: [[1,2], [3,4], [5,6]]
    // Batch 1: [[7,8], [9,10], [11,12]]

    // Select along dim 1 (middle dimension)
    let b = a.index_select(&[0, 2], 1)?;
    assert_eq!(b.dims(), &[2, 2, 2]);
    // Batch 0: [[1,2], [5,6]]
    // Batch 1: [[7,8], [11,12]]
    assert_eq!(b.to_vec()?, vec![1., 2., 5., 6., 7., 8., 11., 12.]);
    Ok(())
}
test_both_backends!(test_index_select_3d, test_index_select_3d_impl);

// =============================================================================
// Reduce tests
// =============================================================================

fn test_max_dim0_impl<B: Backend>(dev: &B) -> Result<()> {
    // 3x4 tensor, max along dim 0 -> 4
    let a: Tensor<f32, B> =
        Tensor::from_vec(vec![1., 5., 3., 4., 8., 2., 7., 6., 9., 0., 1., 2.], (3, 4), dev)?;
    // Column-wise max:
    // col 0: max(1, 8, 9) = 9
    // col 1: max(5, 2, 0) = 5
    // col 2: max(3, 7, 1) = 7
    // col 3: max(4, 6, 2) = 6
    let b = a.max(0)?;
    assert_eq!(b.dims(), &[4]);
    assert_eq!(b.to_vec()?, vec![9., 5., 7., 6.]);
    Ok(())
}
test_both_backends!(test_max_dim0, test_max_dim0_impl);

fn test_max_dim1_impl<B: Backend>(dev: &B) -> Result<()> {
    // 3x4 tensor, max along dim 1 -> 3
    let a: Tensor<f32, B> =
        Tensor::from_vec(vec![1., 5., 3., 4., 8., 2., 7., 6., 9., 0., 1., 2.], (3, 4), dev)?;
    // Row-wise max:
    // row 0: max(1, 5, 3, 4) = 5
    // row 1: max(8, 2, 7, 6) = 8
    // row 2: max(9, 0, 1, 2) = 9
    let b = a.max(1)?;
    assert_eq!(b.dims(), &[3]);
    assert_eq!(b.to_vec()?, vec![5., 8., 9.]);
    Ok(())
}
test_both_backends!(test_max_dim1, test_max_dim1_impl);

fn test_min_dim0_impl<B: Backend>(dev: &B) -> Result<()> {
    // 3x4 tensor, min along dim 0 -> 4
    let a: Tensor<f32, B> =
        Tensor::from_vec(vec![1., 5., 3., 4., 8., 2., 7., 6., 9., 0., 1., 2.], (3, 4), dev)?;
    // Column-wise min:
    // col 0: min(1, 8, 9) = 1
    // col 1: min(5, 2, 0) = 0
    // col 2: min(3, 7, 1) = 1
    // col 3: min(4, 6, 2) = 2
    let b = a.min(0)?;
    assert_eq!(b.dims(), &[4]);
    assert_eq!(b.to_vec()?, vec![1., 0., 1., 2.]);
    Ok(())
}
test_both_backends!(test_min_dim0, test_min_dim0_impl);

fn test_min_dim1_impl<B: Backend>(dev: &B) -> Result<()> {
    // 3x4 tensor, min along dim 1 -> 3
    let a: Tensor<f32, B> =
        Tensor::from_vec(vec![1., 5., 3., 4., 8., 2., 7., 6., 9., 0., 1., 2.], (3, 4), dev)?;
    // Row-wise min:
    // row 0: min(1, 5, 3, 4) = 1
    // row 1: min(8, 2, 7, 6) = 2
    // row 2: min(9, 0, 1, 2) = 0
    let b = a.min(1)?;
    assert_eq!(b.dims(), &[3]);
    assert_eq!(b.to_vec()?, vec![1., 2., 0.]);
    Ok(())
}
test_both_backends!(test_min_dim1, test_min_dim1_impl);

fn test_argmin_dim0_impl<B: Backend>(dev: &B) -> Result<()> {
    // 3x4 tensor, argmin along dim 0 -> 4
    let a: Tensor<f32, B> =
        Tensor::from_vec(vec![1., 5., 3., 4., 8., 2., 7., 6., 9., 0., 1., 2.], (3, 4), dev)?;
    // Column-wise argmin:
    // col 0: argmin(1, 8, 9) = 0
    // col 1: argmin(5, 2, 0) = 2
    // col 2: argmin(3, 7, 1) = 2
    // col 3: argmin(4, 6, 2) = 2
    let b = a.argmin(0)?;
    assert_eq!(b.dims(), &[4]);
    assert_eq!(b.to_vec()?, vec![0i64, 2, 2, 2]);
    Ok(())
}
test_both_backends!(test_argmin_dim0, test_argmin_dim0_impl);

fn test_argmin_dim1_impl<B: Backend>(dev: &B) -> Result<()> {
    // 3x4 tensor, argmin along dim 1 -> 3
    let a: Tensor<f32, B> =
        Tensor::from_vec(vec![1., 5., 3., 4., 8., 2., 7., 6., 9., 0., 1., 2.], (3, 4), dev)?;
    // Row-wise argmin:
    // row 0: argmin(1, 5, 3, 4) = 0
    // row 1: argmin(8, 2, 7, 6) = 1
    // row 2: argmin(9, 0, 1, 2) = 1
    let b = a.argmin(1)?;
    assert_eq!(b.dims(), &[3]);
    assert_eq!(b.to_vec()?, vec![0i64, 1, 1]);
    Ok(())
}
test_both_backends!(test_argmin_dim1, test_argmin_dim1_impl);

fn test_max_3d_impl<B: Backend>(dev: &B) -> Result<()> {
    // 2x3x2 tensor, max along dim 1 -> 2x2
    let a: Tensor<f32, B> = Tensor::from_vec((1..=12).map(|x| x as f32).collect(), (2, 3, 2), dev)?;
    // Batch 0: [[1,2], [3,4], [5,6]] -> max along rows: [5, 6]
    // Batch 1: [[7,8], [9,10], [11,12]] -> max along rows: [11, 12]
    let b = a.max(1)?;
    assert_eq!(b.dims(), &[2, 2]);
    assert_eq!(b.to_vec()?, vec![5., 6., 11., 12.]);
    Ok(())
}
test_both_backends!(test_max_3d, test_max_3d_impl);

// =============================================================================
// Broadcast tests
// =============================================================================

fn test_broadcast_add_same_shape_impl<B: Backend>(dev: &B) -> Result<()> {
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4.], (2, 2), dev)?;
    let b: Tensor<f32, B> = Tensor::from_vec(vec![10., 20., 30., 40.], (2, 2), dev)?;
    let c = a.broadcast_add(&b)?;
    assert_eq!(c.dims(), &[2, 2]);
    assert_eq!(c.to_vec()?, vec![11., 22., 33., 44.]);
    Ok(())
}
test_both_backends!(test_broadcast_add_same_shape, test_broadcast_add_same_shape_impl);

fn test_broadcast_add_1d_to_2d_impl<B: Backend>(dev: &B) -> Result<()> {
    // [2, 3] + [3] -> [2, 3]
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), dev)?;
    let b: Tensor<f32, B> = Tensor::from_vec(vec![10., 20., 30.], (3,), dev)?;
    let c = a.broadcast_add(&b)?;
    assert_eq!(c.dims(), &[2, 3]);
    // Row 0: [1+10, 2+20, 3+30] = [11, 22, 33]
    // Row 1: [4+10, 5+20, 6+30] = [14, 25, 36]
    assert_eq!(c.to_vec()?, vec![11., 22., 33., 14., 25., 36.]);
    Ok(())
}
test_both_backends!(test_broadcast_add_1d_to_2d, test_broadcast_add_1d_to_2d_impl);

fn test_broadcast_mul_column_impl<B: Backend>(dev: &B) -> Result<()> {
    // [2, 3] * [2, 1] -> [2, 3]
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), dev)?;
    let b: Tensor<f32, B> = Tensor::from_vec(vec![2., 3.], (2, 1), dev)?;
    let c = a.broadcast_mul(&b)?;
    assert_eq!(c.dims(), &[2, 3]);
    // Row 0: [1*2, 2*2, 3*2] = [2, 4, 6]
    // Row 1: [4*3, 5*3, 6*3] = [12, 15, 18]
    assert_eq!(c.to_vec()?, vec![2., 4., 6., 12., 15., 18.]);
    Ok(())
}
test_both_backends!(test_broadcast_mul_column, test_broadcast_mul_column_impl);

fn test_broadcast_sub_impl<B: Backend>(dev: &B) -> Result<()> {
    // [2, 3] - [3] -> [2, 3]
    let a: Tensor<f32, B> = Tensor::from_vec(vec![10., 20., 30., 40., 50., 60.], (2, 3), dev)?;
    let b: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3.], (3,), dev)?;
    let c = a.broadcast_sub(&b)?;
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(c.to_vec()?, vec![9., 18., 27., 39., 48., 57.]);
    Ok(())
}
test_both_backends!(test_broadcast_sub, test_broadcast_sub_impl);

fn test_broadcast_div_impl<B: Backend>(dev: &B) -> Result<()> {
    // [2, 3] / [2, 1] -> [2, 3]
    let a: Tensor<f32, B> = Tensor::from_vec(vec![2., 4., 6., 9., 12., 15.], (2, 3), dev)?;
    let b: Tensor<f32, B> = Tensor::from_vec(vec![2., 3.], (2, 1), dev)?;
    let c = a.broadcast_div(&b)?;
    assert_eq!(c.dims(), &[2, 3]);
    // Row 0: [2/2, 4/2, 6/2] = [1, 2, 3]
    // Row 1: [9/3, 12/3, 15/3] = [3, 4, 5]
    assert_eq!(c.to_vec()?, vec![1., 2., 3., 3., 4., 5.]);
    Ok(())
}
test_both_backends!(test_broadcast_div, test_broadcast_div_impl);

fn test_broadcast_3d_impl<B: Backend>(dev: &B) -> Result<()> {
    // [2, 3, 4] + [4] -> [2, 3, 4]
    let a: Tensor<f32, B> = Tensor::from_vec((1..=24).map(|x| x as f32).collect(), (2, 3, 4), dev)?;
    let b: Tensor<f32, B> = Tensor::from_vec(vec![100., 200., 300., 400.], (4,), dev)?;
    let c = a.broadcast_add(&b)?;
    assert_eq!(c.dims(), &[2, 3, 4]);
    let c_vec = c.to_vec()?;
    // First element: 1 + 100 = 101
    assert_eq!(c_vec[0], 101.);
    // Second element: 2 + 200 = 202
    assert_eq!(c_vec[1], 202.);
    // Fifth element: 5 + 100 = 105
    assert_eq!(c_vec[4], 105.);
    Ok(())
}
test_both_backends!(test_broadcast_3d, test_broadcast_3d_impl);

// =============================================================================
// Unsqueeze tests
// =============================================================================

fn test_unsqueeze_dim0_impl<B: Backend>(dev: &B) -> Result<()> {
    // [3, 4] -> unsqueeze(0) -> [1, 3, 4]
    let a: Tensor<f32, B> = Tensor::from_vec((1..=12).map(|x| x as f32).collect(), (3, 4), dev)?;
    let b = a.unsqueeze(0)?;
    assert_eq!(b.dims(), &[1, 3, 4]);
    assert_eq!(b.to_vec()?, a.to_vec()?);
    Ok(())
}
test_both_backends!(test_unsqueeze_dim0, test_unsqueeze_dim0_impl);

fn test_unsqueeze_dim1_impl<B: Backend>(dev: &B) -> Result<()> {
    // [3, 4] -> unsqueeze(1) -> [3, 1, 4]
    let a: Tensor<f32, B> = Tensor::from_vec((1..=12).map(|x| x as f32).collect(), (3, 4), dev)?;
    let b = a.unsqueeze(1)?;
    assert_eq!(b.dims(), &[3, 1, 4]);
    assert_eq!(b.to_vec()?, a.to_vec()?);
    Ok(())
}
test_both_backends!(test_unsqueeze_dim1, test_unsqueeze_dim1_impl);

fn test_unsqueeze_dim_last_impl<B: Backend>(dev: &B) -> Result<()> {
    // [3, 4] -> unsqueeze(2) -> [3, 4, 1]
    let a: Tensor<f32, B> = Tensor::from_vec((1..=12).map(|x| x as f32).collect(), (3, 4), dev)?;
    let b = a.unsqueeze(2)?;
    assert_eq!(b.dims(), &[3, 4, 1]);
    assert_eq!(b.to_vec()?, a.to_vec()?);
    Ok(())
}
test_both_backends!(test_unsqueeze_dim_last, test_unsqueeze_dim_last_impl);

fn test_unsqueeze_1d_impl<B: Backend>(dev: &B) -> Result<()> {
    // [4] -> unsqueeze(0) -> [1, 4]
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4.], (4,), dev)?;
    let b = a.unsqueeze(0)?;
    assert_eq!(b.dims(), &[1, 4]);

    // [4] -> unsqueeze(1) -> [4, 1]
    let c = a.unsqueeze(1)?;
    assert_eq!(c.dims(), &[4, 1]);
    Ok(())
}
test_both_backends!(test_unsqueeze_1d, test_unsqueeze_1d_impl);

// =============================================================================
// Pad with zeros tests
// =============================================================================

fn test_pad_with_zeros_1d_impl<B: Backend>(dev: &B) -> Result<()> {
    // [4] -> pad_with_zeros(0, 2, 3) -> [9]
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4.], (4,), dev)?;
    let b = a.pad_with_zeros(0, 2, 3)?;
    assert_eq!(b.dims(), &[9]);
    assert_eq!(b.to_vec()?, vec![0., 0., 1., 2., 3., 4., 0., 0., 0.]);
    Ok(())
}
test_both_backends!(test_pad_with_zeros_1d, test_pad_with_zeros_1d_impl);

fn test_pad_with_zeros_2d_dim0_impl<B: Backend>(dev: &B) -> Result<()> {
    // [2, 3] -> pad_with_zeros(0, 1, 1) -> [4, 3]
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), dev)?;
    let b = a.pad_with_zeros(0, 1, 1)?;
    assert_eq!(b.dims(), &[4, 3]);
    // Row 0: zeros, Row 1: [1,2,3], Row 2: [4,5,6], Row 3: zeros
    assert_eq!(b.to_vec()?, vec![0., 0., 0., 1., 2., 3., 4., 5., 6., 0., 0., 0.]);
    Ok(())
}
test_both_backends!(test_pad_with_zeros_2d_dim0, test_pad_with_zeros_2d_dim0_impl);

fn test_pad_with_zeros_2d_dim1_impl<B: Backend>(dev: &B) -> Result<()> {
    // [2, 3] -> pad_with_zeros(1, 1, 2) -> [2, 6]
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), dev)?;
    let b = a.pad_with_zeros(1, 1, 2)?;
    assert_eq!(b.dims(), &[2, 6]);
    // Row 0: [0, 1, 2, 3, 0, 0]
    // Row 1: [0, 4, 5, 6, 0, 0]
    assert_eq!(b.to_vec()?, vec![0., 1., 2., 3., 0., 0., 0., 4., 5., 6., 0., 0.]);
    Ok(())
}
test_both_backends!(test_pad_with_zeros_2d_dim1, test_pad_with_zeros_2d_dim1_impl);

fn test_pad_with_zeros_3d_impl<B: Backend>(dev: &B) -> Result<()> {
    // [2, 2, 3] -> pad_with_zeros(1, 1, 0) -> [2, 3, 3]
    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let a: Tensor<f32, B> = Tensor::from_vec(data, (2, 2, 3), dev)?;
    let b = a.pad_with_zeros(1, 1, 0)?;
    assert_eq!(b.dims(), &[2, 3, 3]);
    // First batch: [[0,0,0], [1,2,3], [4,5,6]]
    // Second batch: [[0,0,0], [7,8,9], [10,11,12]]
    assert_eq!(
        b.to_vec()?,
        vec![0., 0., 0., 1., 2., 3., 4., 5., 6., 0., 0., 0., 7., 8., 9., 10., 11., 12.]
    );
    Ok(())
}
test_both_backends!(test_pad_with_zeros_3d, test_pad_with_zeros_3d_impl);

// =============================================================================
// Conv1d tests
// =============================================================================

fn test_conv1d_simple_impl<B: Backend>(dev: &B) -> Result<()> {
    // Input: (batch=1, in_channels=1, length=5)
    // Kernel: (out_channels=1, in_channels=1, kernel_size=3)
    // No padding, stride=1, groups=1
    let input: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5.], (1, 1, 5), dev)?;
    let kernel: Tensor<f32, B> = Tensor::from_vec(vec![1., 0., -1.], (1, 1, 3), dev)?;

    let output = input.conv1d(&kernel, None, 1, 0, 1, 1)?;
    assert_eq!(output.dims(), &[1, 1, 3]);
    // output[i] = input[i]*1 + input[i+1]*0 + input[i+2]*(-1)
    // output[0] = 1 - 3 = -2
    // output[1] = 2 - 4 = -2
    // output[2] = 3 - 5 = -2
    assert_eq!(output.to_vec()?, vec![-2., -2., -2.]);
    Ok(())
}
test_both_backends!(test_conv1d_simple, test_conv1d_simple_impl);

fn test_conv1d_with_padding_impl<B: Backend>(dev: &B) -> Result<()> {
    // Input: (batch=1, in_channels=1, length=4)
    // Kernel: (out_channels=1, in_channels=1, kernel_size=3)
    // Padding=1, stride=1
    let input: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4.], (1, 1, 4), dev)?;
    let kernel: Tensor<f32, B> = Tensor::from_vec(vec![1., 1., 1.], (1, 1, 3), dev)?;

    let output = input.conv1d(&kernel, None, 1, 1, 1, 1)?;
    assert_eq!(output.dims(), &[1, 1, 4]);
    // With padding=1, we have [0, 1, 2, 3, 4, 0] as effective input
    // output[0] = 0 + 1 + 2 = 3
    // output[1] = 1 + 2 + 3 = 6
    // output[2] = 2 + 3 + 4 = 9
    // output[3] = 3 + 4 + 0 = 7
    assert_eq!(output.to_vec()?, vec![3., 6., 9., 7.]);
    Ok(())
}
test_both_backends!(test_conv1d_with_padding, test_conv1d_with_padding_impl);

fn test_conv1d_with_stride_impl<B: Backend>(dev: &B) -> Result<()> {
    // Input: (batch=1, in_channels=1, length=6)
    // Kernel: (out_channels=1, in_channels=1, kernel_size=2)
    // Stride=2
    let input: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (1, 1, 6), dev)?;
    let kernel: Tensor<f32, B> = Tensor::from_vec(vec![1., 1.], (1, 1, 2), dev)?;

    let output = input.conv1d(&kernel, None, 2, 0, 1, 1)?;
    // out_length = (6 - 2) / 2 + 1 = 3
    assert_eq!(output.dims(), &[1, 1, 3]);
    // output[0] = 1 + 2 = 3
    // output[1] = 3 + 4 = 7
    // output[2] = 5 + 6 = 11
    assert_eq!(output.to_vec()?, vec![3., 7., 11.]);
    Ok(())
}
test_both_backends!(test_conv1d_with_stride, test_conv1d_with_stride_impl);

fn test_conv1d_with_bias_impl<B: Backend>(dev: &B) -> Result<()> {
    let input: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5.], (1, 1, 5), dev)?;
    let kernel: Tensor<f32, B> = Tensor::from_vec(vec![1., 1., 1.], (1, 1, 3), dev)?;
    let bias: Tensor<f32, B> = Tensor::from_vec(vec![10.], (1,), dev)?;

    let output = input.conv1d(&kernel, Some(&bias), 1, 0, 1, 1)?;
    assert_eq!(output.dims(), &[1, 1, 3]);
    // Without bias: [6, 9, 12], with bias: [16, 19, 22]
    assert_eq!(output.to_vec()?, vec![16., 19., 22.]);
    Ok(())
}
test_both_backends!(test_conv1d_with_bias, test_conv1d_with_bias_impl);

fn test_conv1d_multi_channel_impl<B: Backend>(dev: &B) -> Result<()> {
    // Input: (batch=1, in_channels=2, length=3)
    // Kernel: (out_channels=2, in_channels=2, kernel_size=2)
    let input: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (1, 2, 3), dev)?;
    // kernel[0] for out_channel 0: [[1,1], [0,0]] - only uses in_channel 0
    // kernel[1] for out_channel 1: [[0,0], [1,1]] - only uses in_channel 1
    let kernel: Tensor<f32, B> =
        Tensor::from_vec(vec![1., 1., 0., 0., 0., 0., 1., 1.], (2, 2, 2), dev)?;

    let output = input.conv1d(&kernel, None, 1, 0, 1, 1)?;
    assert_eq!(output.dims(), &[1, 2, 2]);
    // out[0,0] = 1+2 = 3, out[0,1] = 2+3 = 5
    // out[1,0] = 4+5 = 9, out[1,1] = 5+6 = 11
    assert_eq!(output.to_vec()?, vec![3., 5., 9., 11.]);
    Ok(())
}
test_both_backends!(test_conv1d_multi_channel, test_conv1d_multi_channel_impl);

// =============================================================================
// Conv transpose 1d tests
// =============================================================================

fn test_conv_transpose1d_simple_impl<B: Backend>(dev: &B) -> Result<()> {
    // Input: (batch=1, in_channels=1, length=3)
    // Kernel: (in_channels=1, out_channels=1, kernel_size=3)
    // stride=1, no padding
    let input: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3.], (1, 1, 3), dev)?;
    let kernel: Tensor<f32, B> = Tensor::from_vec(vec![1., 1., 1.], (1, 1, 3), dev)?;

    let output = input.conv_transpose1d(&kernel, None, 1, 0, 0, 1)?;
    // out_length = (3-1)*1 + 3 + 0 - 0 = 5
    assert_eq!(output.dims(), &[1, 1, 5]);
    // Each input value contributes to 3 consecutive output positions
    // output[0] = 1*1 = 1
    // output[1] = 1*1 + 2*1 = 3
    // output[2] = 1*1 + 2*1 + 3*1 = 6
    // output[3] = 2*1 + 3*1 = 5
    // output[4] = 3*1 = 3
    assert_eq!(output.to_vec()?, vec![1., 3., 6., 5., 3.]);
    Ok(())
}
test_both_backends!(test_conv_transpose1d_simple, test_conv_transpose1d_simple_impl);

fn test_conv_transpose1d_with_stride_impl<B: Backend>(dev: &B) -> Result<()> {
    // Input: (batch=1, in_channels=1, length=3)
    // Kernel: (in_channels=1, out_channels=1, kernel_size=2)
    // stride=2
    let input: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3.], (1, 1, 3), dev)?;
    let kernel: Tensor<f32, B> = Tensor::from_vec(vec![1., 1.], (1, 1, 2), dev)?;

    let output = input.conv_transpose1d(&kernel, None, 2, 0, 0, 1)?;
    // out_length = (3-1)*2 + 2 + 0 - 0 = 6
    assert_eq!(output.dims(), &[1, 1, 6]);
    // Input at position i contributes to output positions i*stride + k
    // input[0]=1 -> output[0], output[1]
    // input[1]=2 -> output[2], output[3]
    // input[2]=3 -> output[4], output[5]
    assert_eq!(output.to_vec()?, vec![1., 1., 2., 2., 3., 3.]);
    Ok(())
}
test_both_backends!(test_conv_transpose1d_with_stride, test_conv_transpose1d_with_stride_impl);

fn test_conv_transpose1d_with_bias_impl<B: Backend>(dev: &B) -> Result<()> {
    let input: Tensor<f32, B> = Tensor::from_vec(vec![1., 2.], (1, 1, 2), dev)?;
    let kernel: Tensor<f32, B> = Tensor::from_vec(vec![1., 1.], (1, 1, 2), dev)?;
    let bias: Tensor<f32, B> = Tensor::from_vec(vec![5.], (1,), dev)?;

    let output = input.conv_transpose1d(&kernel, Some(&bias), 1, 0, 0, 1)?;
    // out_length = (2-1)*1 + 2 = 3
    assert_eq!(output.dims(), &[1, 1, 3]);
    // Without bias: [1, 3, 2], with bias: [6, 8, 7]
    assert_eq!(output.to_vec()?, vec![6., 8., 7.]);
    Ok(())
}
test_both_backends!(test_conv_transpose1d_with_bias, test_conv_transpose1d_with_bias_impl);

// =============================================================================
// Pad with same tests
// =============================================================================

fn test_pad_with_same_1d_impl<B: Backend>(dev: &B) -> Result<()> {
    // [1, 2, 3, 4] -> pad_with_same(0, 2, 3) -> [1, 1, 1, 2, 3, 4, 4, 4, 4]
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4.], (4,), dev)?;
    let b = a.pad_with_same(0, 2, 3)?;
    assert_eq!(b.dims(), &[9]);
    assert_eq!(b.to_vec()?, vec![1., 1., 1., 2., 3., 4., 4., 4., 4.]);
    Ok(())
}
test_both_backends!(test_pad_with_same_1d, test_pad_with_same_1d_impl);

fn test_pad_with_same_2d_dim0_impl<B: Backend>(dev: &B) -> Result<()> {
    // [2, 3] -> pad_with_same(0, 1, 1) -> [4, 3]
    // Replicates first and last rows
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), dev)?;
    let b = a.pad_with_same(0, 1, 1)?;
    assert_eq!(b.dims(), &[4, 3]);
    // Row 0: copy of row 0 = [1,2,3]
    // Row 1: original row 0 = [1,2,3]
    // Row 2: original row 1 = [4,5,6]
    // Row 3: copy of row 1 = [4,5,6]
    assert_eq!(b.to_vec()?, vec![1., 2., 3., 1., 2., 3., 4., 5., 6., 4., 5., 6.]);
    Ok(())
}
test_both_backends!(test_pad_with_same_2d_dim0, test_pad_with_same_2d_dim0_impl);

fn test_pad_with_same_2d_dim1_impl<B: Backend>(dev: &B) -> Result<()> {
    // [2, 3] -> pad_with_same(1, 1, 2) -> [2, 6]
    // Replicates first and last columns
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), dev)?;
    let b = a.pad_with_same(1, 1, 2)?;
    assert_eq!(b.dims(), &[2, 6]);
    // Row 0: [1, 1, 2, 3, 3, 3]
    // Row 1: [4, 4, 5, 6, 6, 6]
    assert_eq!(b.to_vec()?, vec![1., 1., 2., 3., 3., 3., 4., 4., 5., 6., 6., 6.]);
    Ok(())
}
test_both_backends!(test_pad_with_same_2d_dim1, test_pad_with_same_2d_dim1_impl);

fn test_pad_with_same_3d_impl<B: Backend>(dev: &B) -> Result<()> {
    // [2, 2, 2] -> pad_with_same(1, 1, 1) -> [2, 4, 2]
    let data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let a: Tensor<f32, B> = Tensor::from_vec(data, (2, 2, 2), dev)?;
    let b = a.pad_with_same(1, 1, 1)?;
    assert_eq!(b.dims(), &[2, 4, 2]);
    // First batch [2,2]: [[1,2], [3,4]] -> [[1,2], [1,2], [3,4], [3,4]]
    // Second batch [2,2]: [[5,6], [7,8]] -> [[5,6], [5,6], [7,8], [7,8]]
    assert_eq!(b.to_vec()?, vec![1., 2., 1., 2., 3., 4., 3., 4., 5., 6., 5., 6., 7., 8., 7., 8.]);
    Ok(())
}
test_both_backends!(test_pad_with_same_3d, test_pad_with_same_3d_impl);

// =============================================================================
// Sum keepdim tests
// =============================================================================

fn test_sum_keepdim_1d_impl<B: Backend>(dev: &B) -> Result<()> {
    // [5] -> sum_keepdim(0) -> [1]
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5.], (5,), dev)?;
    let b = a.sum_keepdim(vec![0])?;
    assert_eq!(b.dims(), &[1]);
    assert_eq!(b.to_vec()?, vec![15.]);
    Ok(())
}
test_both_backends!(test_sum_keepdim_1d, test_sum_keepdim_1d_impl);

fn test_sum_keepdim_2d_dim0_impl<B: Backend>(dev: &B) -> Result<()> {
    // [3, 4] -> sum_keepdim(0) -> [1, 4]
    let a: Tensor<f32, B> =
        Tensor::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.], (3, 4), dev)?;
    let b = a.sum_keepdim(vec![0])?;
    assert_eq!(b.dims(), &[1, 4]);
    // Column sums: 1+5+9=15, 2+6+10=18, 3+7+11=21, 4+8+12=24
    assert_eq!(b.to_vec()?, vec![15., 18., 21., 24.]);
    Ok(())
}
test_both_backends!(test_sum_keepdim_2d_dim0, test_sum_keepdim_2d_dim0_impl);

fn test_sum_keepdim_2d_dim1_impl<B: Backend>(dev: &B) -> Result<()> {
    // [3, 4] -> sum_keepdim(1) -> [3, 1]
    let a: Tensor<f32, B> =
        Tensor::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.], (3, 4), dev)?;
    let b = a.sum_keepdim(vec![1])?;
    assert_eq!(b.dims(), &[3, 1]);
    // Row sums: 1+2+3+4=10, 5+6+7+8=26, 9+10+11+12=42
    assert_eq!(b.to_vec()?, vec![10., 26., 42.]);
    Ok(())
}
test_both_backends!(test_sum_keepdim_2d_dim1, test_sum_keepdim_2d_dim1_impl);

fn test_sum_keepdim_3d_impl<B: Backend>(dev: &B) -> Result<()> {
    // [2, 3, 2] -> sum_keepdim(1) -> [2, 1, 2]
    let a: Tensor<f32, B> = Tensor::from_vec((1..=12).map(|x| x as f32).collect(), (2, 3, 2), dev)?;
    let b = a.sum_keepdim(vec![1])?;
    assert_eq!(b.dims(), &[2, 1, 2]);
    // Batch 0: [[1,2], [3,4], [5,6]] -> sum along dim 1 -> [9, 12]
    // Batch 1: [[7,8], [9,10], [11,12]] -> sum along dim 1 -> [27, 30]
    assert_eq!(b.to_vec()?, vec![9., 12., 27., 30.]);
    Ok(())
}
test_both_backends!(test_sum_keepdim_3d, test_sum_keepdim_3d_impl);

fn test_sum_keepdim_multiple_dims_impl<B: Backend>(dev: &B) -> Result<()> {
    // [2, 3, 4] -> sum_keepdim([1, 2]) -> [2, 1, 1]
    let a: Tensor<f32, B> = Tensor::from_vec((1..=24).map(|x| x as f32).collect(), (2, 3, 4), dev)?;
    let b = a.sum_keepdim(vec![1, 2])?;
    assert_eq!(b.dims(), &[2, 1, 1]);
    // Batch 0: sum of 1..12 = 78
    // Batch 1: sum of 13..24 = 222
    assert_eq!(b.to_vec()?, vec![78., 222.]);
    Ok(())
}
test_both_backends!(test_sum_keepdim_multiple_dims, test_sum_keepdim_multiple_dims_impl);

// =============================================================================
// Slice set tests
// =============================================================================

fn test_slice_set_dim0_impl<B: Backend>(dev: &B) -> Result<()> {
    // dst: [4, 3], src: [2, 3], set at offset 1 along dim 0
    let dst: Tensor<f32, B> = Tensor::zeros((4, 3), dev)?;
    let src: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), dev)?;

    dst.slice_set(&src, 0, 1)?;
    // Row 0: [0, 0, 0]
    // Row 1: [1, 2, 3]
    // Row 2: [4, 5, 6]
    // Row 3: [0, 0, 0]
    assert_eq!(dst.to_vec()?, vec![0., 0., 0., 1., 2., 3., 4., 5., 6., 0., 0., 0.]);
    Ok(())
}
test_both_backends!(test_slice_set_dim0, test_slice_set_dim0_impl);

fn test_slice_set_dim1_impl<B: Backend>(dev: &B) -> Result<()> {
    // dst: [2, 6], src: [2, 3], set at offset 2 along dim 1
    let dst: Tensor<f32, B> = Tensor::zeros((2, 6), dev)?;
    let src: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), dev)?;

    dst.slice_set(&src, 1, 2)?;
    // Row 0: [0, 0, 1, 2, 3, 0]
    // Row 1: [0, 0, 4, 5, 6, 0]
    assert_eq!(dst.to_vec()?, vec![0., 0., 1., 2., 3., 0., 0., 0., 4., 5., 6., 0.]);
    Ok(())
}
test_both_backends!(test_slice_set_dim1, test_slice_set_dim1_impl);

fn test_slice_set_3d_impl<B: Backend>(dev: &B) -> Result<()> {
    // dst: [2, 4, 3], src: [2, 2, 3], set at offset 1 along dim 1
    let dst: Tensor<f32, B> = Tensor::zeros((2, 4, 3), dev)?;
    let src: Tensor<f32, B> =
        Tensor::from_vec((1..=12).map(|x| x as f32).collect(), (2, 2, 3), dev)?;

    dst.slice_set(&src, 1, 1)?;
    // Batch 0: [[0,0,0], [1,2,3], [4,5,6], [0,0,0]]
    // Batch 1: [[0,0,0], [7,8,9], [10,11,12], [0,0,0]]
    assert_eq!(
        dst.to_vec()?,
        vec![
            0., 0., 0., 1., 2., 3., 4., 5., 6., 0., 0., 0., // batch 0
            0., 0., 0., 7., 8., 9., 10., 11., 12., 0., 0., 0. // batch 1
        ]
    );
    Ok(())
}
test_both_backends!(test_slice_set_3d, test_slice_set_3d_impl);

fn test_slice_set_at_start_impl<B: Backend>(dev: &B) -> Result<()> {
    // dst: [4, 2], src: [2, 2], set at offset 0
    let dst: Tensor<f32, B> = Tensor::full(9., (4, 2), dev)?;
    let src: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4.], (2, 2), dev)?;

    dst.slice_set(&src, 0, 0)?;
    assert_eq!(dst.to_vec()?, vec![1., 2., 3., 4., 9., 9., 9., 9.]);
    Ok(())
}
test_both_backends!(test_slice_set_at_start, test_slice_set_at_start_impl);

fn test_slice_set_at_end_impl<B: Backend>(dev: &B) -> Result<()> {
    // dst: [4, 2], src: [2, 2], set at offset 2 (at the end)
    let dst: Tensor<f32, B> = Tensor::full(9., (4, 2), dev)?;
    let src: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4.], (2, 2), dev)?;

    dst.slice_set(&src, 0, 2)?;
    assert_eq!(dst.to_vec()?, vec![9., 9., 9., 9., 1., 2., 3., 4.]);
    Ok(())
}
test_both_backends!(test_slice_set_at_end, test_slice_set_at_end_impl);

fn test_slice_set_1d_impl<B: Backend>(dev: &B) -> Result<()> {
    // dst: [8], src: [3], set at offset 2
    let dst: Tensor<f32, B> = Tensor::zeros((8,), dev)?;
    let src: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3.], (3,), dev)?;

    dst.slice_set(&src, 0, 2)?;
    assert_eq!(dst.to_vec()?, vec![0., 0., 1., 2., 3., 0., 0., 0.]);
    Ok(())
}
test_both_backends!(test_slice_set_1d, test_slice_set_1d_impl);

// =============================================================================
// Scatter tests
// =============================================================================

fn test_scatter_dim0_impl<B: Backend>(dev: &B) -> Result<()> {
    // dst: 3x3 zeros, scatter src values into dst along dim 0
    // Each (column, target row) pair is unique to avoid non-determinism with parallel writes.
    let dst: Tensor<f32, B> = Tensor::zeros((3, 3), dev)?;
    let src: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), dev)?;
    let ids: Tensor<i64, B> = Tensor::from_vec(vec![0i64, 2, 1, 2, 0, 0], (2, 3), dev)?;

    let result = dst.scatter(&ids, &src, 0)?;
    assert_eq!(result.dims(), &[3, 3]);
    // Col 0: src[0][0]=1 -> row 0, src[1][0]=4 -> row 2
    // Col 1: src[0][1]=2 -> row 2, src[1][1]=5 -> row 0
    // Col 2: src[0][2]=3 -> row 1, src[1][2]=6 -> row 0
    assert_eq!(result.to_vec()?, vec![1., 5., 6., 0., 0., 3., 4., 2., 0.]);
    Ok(())
}
test_both_backends!(test_scatter_dim0, test_scatter_dim0_impl);

fn test_scatter_dim1_impl<B: Backend>(dev: &B) -> Result<()> {
    // dst: 2x4 zeros, scatter src into dst along dim 1
    let dst: Tensor<f32, B> = Tensor::zeros((2, 4), dev)?;
    let src: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), dev)?;
    let ids: Tensor<i64, B> = Tensor::from_vec(vec![3i64, 0, 1, 2, 3, 0], (2, 3), dev)?;

    let result = dst.scatter(&ids, &src, 1)?;
    assert_eq!(result.dims(), &[2, 4]);
    // Row 0: col3=1, col0=2, col1=3 -> [2, 3, 0, 1]
    // Row 1: col2=4, col3=5, col0=6 -> [6, 0, 4, 5]
    assert_eq!(result.to_vec()?, vec![2., 3., 0., 1., 6., 0., 4., 5.]);
    Ok(())
}
test_both_backends!(test_scatter_dim1, test_scatter_dim1_impl);

fn test_scatter_set_dim0_impl<B: Backend>(dev: &B) -> Result<()> {
    // In-place scatter_set
    let dst: Tensor<f32, B> =
        Tensor::from_vec(vec![10., 20., 30., 40., 50., 60., 70., 80., 90.], (3, 3), dev)?;
    let src: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3.], (1, 3), dev)?;
    let ids: Tensor<i64, B> = Tensor::from_vec(vec![2i64, 0, 1], (1, 3), dev)?;

    dst.scatter_set(&ids, &src, 0)?;
    // Row 2 col 0 = 1, Row 0 col 1 = 2, Row 1 col 2 = 3
    assert_eq!(dst.to_vec()?, vec![10., 2., 30., 40., 50., 3., 1., 80., 90.]);
    Ok(())
}
test_both_backends!(test_scatter_set_dim0, test_scatter_set_dim0_impl);

fn test_scatter_3d_impl<B: Backend>(dev: &B) -> Result<()> {
    // 3D scatter along dim 1
    // dst: (2, 3, 2) zeros
    let dst: Tensor<f32, B> = Tensor::zeros((2, 3, 2), dev)?;
    // src: (2, 1, 2)
    let src: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4.], (2, 1, 2), dev)?;
    // ids: (2, 1, 2) - scatter along dim 1
    let ids: Tensor<i64, B> = Tensor::from_vec(vec![2i64, 0, 1, 2], (2, 1, 2), dev)?;

    let result = dst.scatter(&ids, &src, 1)?;
    assert_eq!(result.dims(), &[2, 3, 2]);
    // Batch 0: dst[0][2][0]=1, dst[0][0][1]=2
    // Batch 1: dst[1][1][0]=3, dst[1][2][1]=4
    assert_eq!(result.to_vec()?, vec![0., 2., 0., 0., 1., 0., 0., 0., 3., 0., 0., 4.]);
    Ok(())
}
test_both_backends!(test_scatter_3d, test_scatter_3d_impl);

// =============================================================================
// Broadcast tests
// =============================================================================

fn test_broadcast_as_add_dim_impl<B: Backend>(dev: &B) -> Result<()> {
    // (3,) -> (2, 3): broadcast by prepending a dimension
    let t: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3.], (3,), dev)?;
    let view = t.broadcast_as((2, 3))?;
    assert_eq!(view.dims(), &[2, 3]);
    let result = view.contiguous()?;
    assert_eq!(result.to_vec()?, vec![1., 2., 3., 1., 2., 3.]);
    Ok(())
}
test_both_backends!(test_broadcast_as_add_dim, test_broadcast_as_add_dim_impl);

fn test_broadcast_as_expand_dim_impl<B: Backend>(dev: &B) -> Result<()> {
    // (2, 1) -> (2, 3): expand dim of size 1
    let t: Tensor<f32, B> = Tensor::from_vec(vec![10., 20.], (2, 1), dev)?;
    let view = t.broadcast_as((2, 3))?;
    assert_eq!(view.dims(), &[2, 3]);
    let result = view.contiguous()?;
    assert_eq!(result.to_vec()?, vec![10., 10., 10., 20., 20., 20.]);
    Ok(())
}
test_both_backends!(test_broadcast_as_expand_dim, test_broadcast_as_expand_dim_impl);

fn test_broadcast_as_3d_impl<B: Backend>(dev: &B) -> Result<()> {
    // (1, 3) -> (2, 4, 3): prepend dim and expand dim 0
    let t: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3.], (1, 3), dev)?;
    let view = t.broadcast_as((2, 4, 3))?;
    assert_eq!(view.dims(), &[2, 4, 3]);
    let result = view.contiguous()?;
    let expected: Vec<f32> = [1., 2., 3.].repeat(8);
    assert_eq!(result.to_vec()?, expected);
    Ok(())
}
test_both_backends!(test_broadcast_as_3d, test_broadcast_as_3d_impl);

fn test_broadcast_as_noop_impl<B: Backend>(dev: &B) -> Result<()> {
    // (2, 3) -> (2, 3): no-op broadcast
    let t: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), dev)?;
    let view = t.broadcast_as((2, 3))?;
    assert_eq!(view.dims(), &[2, 3]);
    let result = view.contiguous()?;
    assert_eq!(result.to_vec()?, vec![1., 2., 3., 4., 5., 6.]);
    Ok(())
}
test_both_backends!(test_broadcast_as_noop, test_broadcast_as_noop_impl);

fn test_broadcast_as_from_view_impl<B: Backend>(dev: &B) -> Result<()> {
    // TensorView::broadcast_as: narrow then broadcast
    let t: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), dev)?;
    let view: TensorView<f32, B> = TensorView::from(&t);
    let narrowed = view.narrow(0, ..1)?; // (1, 3)
    let broadcast = narrowed.broadcast_as((3, 3))?;
    let result = broadcast.contiguous()?;
    assert_eq!(result.to_vec()?, vec![1., 2., 3., 1., 2., 3., 1., 2., 3.]);
    Ok(())
}
test_both_backends!(test_broadcast_as_from_view, test_broadcast_as_from_view_impl);

fn test_broadcast_as_error_impl<B: Backend>(dev: &B) -> Result<()> {
    // Incompatible shapes should error
    let t: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3.], (3,), dev)?;
    assert!(t.broadcast_as((2, 4)).is_err());
    Ok(())
}
test_both_backends!(test_broadcast_as_error, test_broadcast_as_error_impl);

// =============================================================================
// Matmul with transposed view tests
// =============================================================================

fn test_matmul_transposed_view_impl<B: Backend>(dev: &B) -> Result<()> {
    // Simulate the attention pattern: Q @ K^T where K^T is a transposed TensorView.
    // Shape: Q is (1, 2, 3, 4), K is (1, 2, 3, 4), K^T via transpose(2,3) is (1, 2, 4, 3).
    // Result should be (1, 2, 3, 3).
    let q_data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let k_data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.1).collect();
    let q: Tensor<f32, B> = Tensor::from_vec(q_data, (1, 2, 3, 4), dev)?;
    let k: Tensor<f32, B> = Tensor::from_vec(k_data.clone(), (1, 2, 3, 4), dev)?;

    // Method 1: matmul with a transposed view (zero-copy transpose)
    let k_t_view = k.transpose(2, 3)?;
    let result_view = q.matmul(&k_t_view)?;

    // Method 2: matmul_t with the original K (the old reliable way)
    let k2: Tensor<f32, B> = Tensor::from_vec(k_data, (1, 2, 3, 4), dev)?;
    let result_matmul_t = q.matmul_t(&k2)?;

    // Both should produce the same result.
    let v1 = result_view.to_vec()?;
    let v2 = result_matmul_t.to_vec()?;
    assert_eq!(v1.len(), v2.len());
    for (a, b) in v1.iter().zip(v2.iter()) {
        assert!((a - b).abs() < 1e-4, "mismatch: {a} vs {b}");
    }
    Ok(())
}
test_both_backends!(test_matmul_transposed_view, test_matmul_transposed_view_impl);
