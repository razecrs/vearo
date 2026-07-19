use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write, Result as IoResult};
use vearo_core::Tensor;

/// Saves the given parameters (tensors) to a `.ve` binary weight checkpoint file.
///
/// # Errors
/// Returns an I/O error if the file cannot be created or written to.
#[allow(clippy::cast_possible_truncation)]
pub fn save_checkpoint(params: &[Tensor], path: impl AsRef<std::path::Path>) -> IoResult<()> {
    let mut writer = BufWriter::new(File::create(path)?);

    // Write number of tensors
    writer.write_all(&(params.len() as u32).to_le_bytes())?;

    for tensor in params {
        let dims = tensor.shape().dims();
        writer.write_all(&(dims.len() as u32).to_le_bytes())?;
        for &dim in dims {
            writer.write_all(&(dim as u32).to_le_bytes())?;
        }

        let data = tensor.to_vec_f32();
        writer.write_all(&(data.len() as u32).to_le_bytes())?;
        for &val in &data {
            writer.write_all(&val.to_le_bytes())?;
        }
    }

    writer.flush()?;
    Ok(())
}

/// Loads weights from a `.ve` binary weight checkpoint file into the given parameters.
///
/// # Errors
/// Returns an I/O error if the file cannot be opened or read.
///
/// # Panics
/// Panics if the checkpoint's tensor count or shapes do not match the expected parameters.
#[allow(clippy::cast_possible_truncation)]
pub fn load_checkpoint(params: &[Tensor], path: impl AsRef<std::path::Path>) -> IoResult<()> {
    let mut reader = BufReader::new(File::open(path)?);

    let mut num_tensors_bytes = [0u8; 4];
    reader.read_exact(&mut num_tensors_bytes)?;
    let num_tensors = u32::from_le_bytes(num_tensors_bytes) as usize;

    assert_eq!(
        num_tensors,
        params.len(),
        "Checkpoint tensor count does not match model parameter count"
    );

    for tensor in params {
        let mut dims_len_bytes = [0u8; 4];
        reader.read_exact(&mut dims_len_bytes)?;
        let dims_len = u32::from_le_bytes(dims_len_bytes) as usize;

        let mut dims = Vec::with_capacity(dims_len);
        for _ in 0..dims_len {
            let mut dim_bytes = [0u8; 4];
            reader.read_exact(&mut dim_bytes)?;
            dims.push(u32::from_le_bytes(dim_bytes) as usize);
        }

        assert_eq!(
            dims,
            tensor.shape().dims(),
            "Checkpoint shape does not match parameter shape"
        );

        let mut data_len_bytes = [0u8; 4];
        reader.read_exact(&mut data_len_bytes)?;
        let data_len = u32::from_le_bytes(data_len_bytes) as usize;

        let mut data = vec![0.0f32; data_len];
        for val in &mut data {
            let mut val_bytes = [0u8; 4];
            reader.read_exact(&mut val_bytes)?;
            *val = f32::from_le_bytes(val_bytes);
        }

        tensor.copy_from_slice(&data);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use vearo_core::Device;

    #[test]
    fn test_checkpoint_save_load() {
        vearo_backend_cpu::init();

        let t1 = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], [2, 2]).to(Device::Cpu);
        let t2 = Tensor::from_f32(&[-0.5, 0.5], [2]).to(Device::Cpu);
        let params = vec![t1, t2];

        // Save
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("model_checkpoint.ve");
        save_checkpoint(&params, &path).unwrap();

        // Overwrite params with zeros
        params[0].copy_from_slice(&[0.0, 0.0, 0.0, 0.0]);
        params[1].copy_from_slice(&[0.0, 0.0]);

        // Load
        load_checkpoint(&params, &path).unwrap();

        // Verify values
        assert_eq!(params[0].to_vec_f32(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(params[1].to_vec_f32(), &[-0.5, 0.5]);

        let _ = std::fs::remove_file(path);
    }
}
