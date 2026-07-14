//! Verification harness integration tests against `NumPy` ground truth.

use serde::Deserialize;
use vearo::{Shape, Tensor, init};

#[derive(Deserialize, Debug)]
struct OracleTensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

#[derive(Deserialize, Debug)]
struct OracleTestCase {
    op: String,
    inputs: Vec<OracleTensor>,
    output: OracleTensor,
}

fn assert_tensor_close(got: &Tensor, expected_shape: &[usize], expected_data: &[f32]) {
    assert_eq!(got.shape().dims(), expected_shape);

    let contiguous_got = got.contiguous();
    let shard = contiguous_got.storage_id().shard_idx as usize;
    let slot_idx = contiguous_got.storage_id().slot_idx as usize;
    // Copy the data out under the lock (temporary guard drops at the match end),
    // then compare unlocked.
    let got_vec = match &vearo::core::get_cpu_shard(shard).lock().unwrap().slots[slot_idx]
        .as_ref()
        .expect("Output slot was empty")
        .storage
    {
        vearo::core::CpuStorage::F32(v) => v.clone(),
        _ => panic!("Expected F32 storage"),
    };

    assert_eq!(got_vec.len(), expected_data.len());
    for (i, (&g, &e)) in got_vec.iter().zip(expected_data.iter()).enumerate() {
        assert!(
            (g - e).abs() <= 1e-5,
            "Mismatch at index {i}: got {g}, expected {e}"
        );
    }
}

#[test]
fn test_numpy_oracle_parity() {
    init();

    let oracles_json = include_str!("../../../test_data/oracles.json");
    let cases: Vec<OracleTestCase> = serde_json::from_str(oracles_json).unwrap();

    for case in cases {
        match case.op.as_str() {
            "add" | "add_transposed" | "add_broadcasted" => {
                let x = Tensor::from_f32(&case.inputs[0].data, case.inputs[0].shape.clone());
                let y = Tensor::from_f32(&case.inputs[1].data, case.inputs[1].shape.clone());
                let got = x.add(&y);
                assert_tensor_close(&got, &case.output.shape, &case.output.data);
            }
            "sub" => {
                let x = Tensor::from_f32(&case.inputs[0].data, case.inputs[0].shape.clone());
                let y = Tensor::from_f32(&case.inputs[1].data, case.inputs[1].shape.clone());
                let got = x.sub(&y);
                assert_tensor_close(&got, &case.output.shape, &case.output.data);
            }
            "mul" => {
                let x = Tensor::from_f32(&case.inputs[0].data, case.inputs[0].shape.clone());
                let y = Tensor::from_f32(&case.inputs[1].data, case.inputs[1].shape.clone());
                let got = x.mul(&y);
                assert_tensor_close(&got, &case.output.shape, &case.output.data);
            }
            "div" => {
                let x = Tensor::from_f32(&case.inputs[0].data, case.inputs[0].shape.clone());
                let y = Tensor::from_f32(&case.inputs[1].data, case.inputs[1].shape.clone());
                let got = x.div(&y);
                assert_tensor_close(&got, &case.output.shape, &case.output.data);
            }
            "matmul_2d" | "matmul_batched" => {
                let x = Tensor::from_f32(&case.inputs[0].data, case.inputs[0].shape.clone());
                let y = Tensor::from_f32(&case.inputs[1].data, case.inputs[1].shape.clone());
                let got = x.matmul(&y);
                assert_tensor_close(&got, &case.output.shape, &case.output.data);
            }
            "reshape" => {
                let x = Tensor::from_f32(&case.inputs[0].data, case.inputs[0].shape.clone());
                let out_shape: Shape = case.output.shape.clone().into();
                let got = x.reshape(out_shape);
                assert_tensor_close(&got, &case.output.shape, &case.output.data);
            }
            "transpose" => {
                let x = Tensor::from_f32(&case.inputs[0].data, case.inputs[0].shape.clone());
                let got = x.transpose(0, 1);
                assert_tensor_close(&got, &case.output.shape, &case.output.data);
            }
            "permute" => {
                let x = Tensor::from_f32(&case.inputs[0].data, case.inputs[0].shape.clone());
                let got = x.permute([2, 0, 1]);
                assert_tensor_close(&got, &case.output.shape, &case.output.data);
            }
            _ => unreachable!("Unknown operation: {}", case.op),
        }
    }
}
