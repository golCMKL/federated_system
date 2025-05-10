use candle_core::{DType, Result as CandleResult, Tensor, D, Module, Device};
use candle_nn::{loss, ops, Optimizer, VarBuilder, VarMap, SGD};
use candle_datasets::vision::Dataset;
use candle_app::{LinearModel, Model};
use tokio::net::{TcpStream, TcpListener};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use base64::Engine;
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::Mutex;
use rand::seq::SliceRandom;
use rand::thread_rng;

macro_rules! log_info {
    ($($arg:tt)*) => (println!("[INFO] {}", format!($($arg)*)));
}
macro_rules! log_warn {
    ($($arg:tt)*) => (println!("[WARN] {}", format!($($arg)*)));
}
macro_rules! log_error {
    ($($arg:tt)*) => (eprintln!("[ERROR] {}", format!($($arg)*)));
}

struct Client {
    server_addr: String,
    model: Option<(LinearModel, VarMap, String)>,
    dataset: Arc<Dataset>,
    local_addr: String,
}

impl Client {
    fn new(server_addr: &str) -> Self {
        let full_dataset = candle_datasets::vision::mnist::load().expect("Failed to load MNIST dataset");
        
        let mut rng = thread_rng();
        let mut indices: Vec<usize> = (0..full_dataset.train_images.dim(0).unwrap()).collect();
        indices.shuffle(&mut rng);
        let selected_indices = &indices[..10_000];

        let index_tensor = Tensor::from_vec(
            selected_indices.iter().map(|&i| i as i64).collect::<Vec<i64>>(),
            (10_000,),
            &Device::Cpu,
        ).unwrap();

        let train_images = full_dataset.train_images.index_select(&index_tensor, 0).unwrap();
        let train_labels = full_dataset.train_labels.index_select(&index_tensor, 0).unwrap();

        let dataset = Dataset {
            train_images,
            train_labels,
            test_images: full_dataset.test_images.clone(),
            test_labels: full_dataset.test_labels.clone(),
            labels: full_dataset.labels,
        };

        Client {
            server_addr: server_addr.to_string(),
            model: None,
            dataset: Arc::new(dataset),
            local_addr: String::new(),
        }
    }

    async fn join(&mut self, server_ip: &str, _model: &str) -> Result<(TcpStream, TcpListener)> {
        let mut stream = TcpStream::connect(server_ip).await?;
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let local_addr = listener.local_addr()?.to_string();
        self.local_addr = local_addr.clone();

        let message = format!("REGISTER|{}", local_addr);
        stream.write_all(message.as_bytes()).await?;
        stream.flush().await?;

        let mut buffer = [0; 1024];
        let n = stream.read(&mut buffer).await?;
        let response = String::from_utf8_lossy(&buffer[..n]);
        log_info!("Server response: {}", response);

        stream.write_all(b"READY").await?;
        stream.flush().await?;

        Ok((stream, listener))
    }

    async fn train(&mut self, model_name: &str, epochs: usize) -> CandleResult<()> {
        if let Some((model, varmap, status)) = &mut self.model {
            if *status != "initialized" && *status != "ready" {
                log_warn!("Client model {} already training or invalid state", model_name);
                return Ok(());
            }
            *status = "training".to_string();

            let dev = Device::Cpu;
            let train_images = self.dataset.train_images.to_device(&dev)?;
            let train_labels = self.dataset.train_labels.to_dtype(DType::U32)?.to_device(&dev)?;
            let mut sgd = SGD::new(varmap.all_vars(), 0.1)?;

            let test_images = self.dataset.test_images.to_device(&dev)?;
            let test_labels = self.dataset.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;

            for epoch in 1..=epochs {
                let logits = Module::forward(model, &train_images)?;
                let log_sm = ops::log_softmax(&logits, D::Minus1)?;
                let loss = loss::nll(&log_sm, &train_labels)?;
                sgd.backward_step(&loss)?;

                let test_logits = Module::forward(model, &test_images)?;
                let sum_ok = test_logits
                    .argmax(D::Minus1)?
                    .eq(&test_labels)?
                    .to_dtype(DType::F32)?
                    .sum_all()?
                    .to_scalar::<f32>()?;
                let accuracy = sum_ok / test_labels.dims1()? as f32;

                log_info!(
                    "Client trained: Epoch {:2} | Model: {} | Loss: {:.4} | Accuracy: {:.2}%",
                    epoch,
                    model_name,
                    loss.to_scalar::<f32>()?,
                    accuracy * 100.0
                );                
            }

            *status = "ready".to_string();
            log_info!("Training completed for model: {}", model_name);
        }
        Ok(())
    }

    fn get(&self, model_name: &str) -> Option<(Vec<f32>, Vec<f32>, String)> {
        if let Some((model, _, status)) = &self.model {
            if model_name != "mnist" {
                return None;
            }
            let weights_data = model.weight().ok()?.to_vec2::<f32>().ok()?.into_iter().flatten().collect::<Vec<f32>>();
            let bias_data = model.bias().ok()?.to_vec1::<f32>().ok()?;
            Some((weights_data, bias_data, status.clone()))
        } else {
            None
        }
    }

    fn test(&self, model_name: &str) -> CandleResult<f32> {
        if let Some((model, _, _)) = &self.model {
            if model_name != "mnist" {
                return Err(candle_core::Error::Msg("Model not found".into()));
            }
            let dev = Device::Cpu;
            let test_images = self.dataset.test_images.to_device(&dev)?;
            let test_labels = self.dataset.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;
            let logits = Module::forward(model, &test_images)?;
            let sum_ok = logits
                .argmax(D::Minus1)?
                .eq(&test_labels)?
                .to_dtype(DType::F32)?
                .sum_all()?
                .to_scalar::<f32>()?;
            let accuracy = sum_ok / test_labels.dims1()? as f32;
            Ok(accuracy)
        } else {
            Err(candle_core::Error::Msg("No model available".into()))
        }
    }

    async fn run_inner(listener: TcpListener, client: Arc<Mutex<Self>>) -> Result<()> {
        log_info!("Client listener active at: {}", listener.local_addr()?);

        loop {
            let (mut client_stream, _) = listener.accept().await?;
            let mut buffer = [0; 65536];
            match client_stream.read(&mut buffer).await {
                Ok(n) => {
                    let message = String::from_utf8_lossy(&buffer[..n]);
                    let parts: Vec<&str> = message.split('|').collect();

                    let mut client_guard = client.lock().await;
                    match parts[0] {
                        "TRAIN" if parts.len() == 5 => {
                            log_info!("TRAIN cmd received: model={}, epochs={}", parts[1], parts[4]);
                            let weights_data: Vec<f32> = bincode::deserialize(
                                &base64::engine::general_purpose::STANDARD.decode(parts[2])?,
                            )?;
                            let bias_data: Vec<f32> = bincode::deserialize(
                                &base64::engine::general_purpose::STANDARD.decode(parts[3])?,
                            )?;
                            let epochs: usize = parts[4].parse()?;

                            let weights = Tensor::from_vec(weights_data, &[10, 784], &Device::Cpu)?;
                            let bias = Tensor::from_vec(bias_data, &[10], &Device::Cpu)?;
                            let varmap = VarMap::new();
                            let vs = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
                            let model = LinearModel::new(vs)?;
                            {
                                let mut data = varmap.data().lock().unwrap();
                                data.get_mut("linear.weight").unwrap().set(&weights)?;
                                data.get_mut("linear.bias").unwrap().set(&bias)?;
                            }

                            client_guard.model = Some((model, varmap, "initialized".to_string()));
                            client_guard.train(parts[1], epochs).await?;

                            if let Some((model, _, _)) = &client_guard.model {
                                let weights_data = model.weight()?.to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
                                let bias_data = model.bias()?.to_vec1::<f32>()?;
                                let response = format!(
                                    "UPDATE|{}|{}",
                                    base64::engine::general_purpose::STANDARD.encode(&bincode::serialize(&weights_data)?),
                                    base64::engine::general_purpose::STANDARD.encode(&bincode::serialize(&bias_data)?)
                                );
                                client_stream.write_all(response.as_bytes()).await?;
                                client_stream.flush().await?;
                            }
                        }
                        "GET" if parts.len() == 2 => {
                            log_info!("GET model info requested for: {}", parts[1]);
                            if let Some((weights_data, bias_data, status)) = client_guard.get(parts[1]) {
                                let weights = bincode::serialize(&weights_data)?;
                                let bias = bincode::serialize(&bias_data)?;
                                let response = format!(
                                    "MODEL|{}|{}|{}",
                                    base64::engine::general_purpose::STANDARD.encode(&weights),
                                    base64::engine::general_purpose::STANDARD.encode(&bias),
                                    status
                                );
                                client_stream.write_all(response.as_bytes()).await?;
                            } else {
                                client_stream.write_all(b"No model available").await?;
                            }
                            client_stream.flush().await?;
                        }
                        "TEST" if parts.len() == 2 => {
                            log_info!("TEST command for model: {}", parts[1]);
                            match client_guard.test(parts[1]) {
                                Ok(accuracy) => {
                                    let response = format!("ACCURACY|{}", accuracy);
                                    client_stream.write_all(response.as_bytes()).await?;
                                }
                                Err(e) => {
                                    client_stream.write_all(format!("Error: {}", e).as_bytes()).await?;
                                }
                            }
                            client_stream.flush().await?;
                        }
                        "COMPLETE" => {
                            log_info!("Server indicated training COMPLETE");
                        }
                        _ => {
                            log_warn!("Unknown message received: {}", message);
                        }
                    }
                }
                Err(e) => log_error!("Error reading from server: {}", e),
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = Arc::new(Mutex::new(Client::new("127.0.0.1:50051")));
    let (_stream, listener) = {
        let mut client_guard = client.lock().await;
        client_guard.join("127.0.0.1:50051", "mnist").await?
    };
    log_info!("Client setup complete on {}", client.lock().await.local_addr);

    let client_clone = Arc::clone(&client);
    tokio::spawn(async move {
        if let Err(e) = Client::run_inner(listener, client_clone).await {
            log_error!("Client run error: {}", e);
        }
    });

    tokio::signal::ctrl_c().await?;
    log_info!("Client terminated.");
    Ok(())
}
