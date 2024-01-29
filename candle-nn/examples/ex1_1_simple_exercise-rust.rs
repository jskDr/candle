#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::{DType, Device, Result, Tensor};
use candle_nn::{linear, AdamW, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};

fn gen_data() -> Result<(Tensor, Tensor)> {
    // Generate some sample linear data.
    let w_gen = Tensor::new(&[[2_f32]], &Device::Cpu)?;
    let b_gen = Tensor::new(1_f32, &Device::Cpu)?;
    let gen = Linear::new(w_gen, Some(b_gen));
    let sample_xs = Tensor::new(&[[0f32], [1.], [2.], [3.], [4.]], &Device::Cpu)?;
    let sample_ys = gen.forward(&sample_xs)?;
    Ok((sample_xs, sample_ys))
}

fn main() -> Result<()> {
    let (sample_xs, sample_ys) = gen_data()?;
    println!("sample_xs = {:?}", sample_xs);
    println!("{:?}", sample_xs.to_vec2::<f32>()?); // note to_vec2
    println!("sample_ys = {:?}", sample_ys);
    println!("{:?}", sample_ys.to_vec2::<f32>()?);

    // Use backprop to run a linear regression between samples and get the coefficients back.
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    let model = linear(1, 1, vb.pp("linear"))?;
    println!("init model: {:?}", model);

    let params = ParamsAdamW {
        lr: 0.1,
        ..Default::default()
    };
    let mut opt = AdamW::new(varmap.all_vars(), params)?;
    for step in 0..=1000 {
        let ys = model.forward(&sample_xs)?;
        let loss = ys.sub(&sample_ys)?.sqr()?.sum_all()?;
        opt.backward_step(&loss)?;
        if step % 100 == 0 {
            println!("{step} {}", loss.to_vec0::<f32>()?);
        }
    }
    println!("trained model: {:?}", model);
    println!("{:?}", model.weight);
    Ok(())
}
