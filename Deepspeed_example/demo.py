from accelerate import Accelerator, DeepSpeedPlugin
import torch
from torch.utils.data import TensorDataset, DataLoader

class SimpleNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    input_dim = 10
    hidden_dim = 256
    output_dim = 2
    batch_size = 64
    data_size = 10000

    # 创建数据时明确指定dtype为float32（PyTorch默认）
    input_data = torch.randn(data_size, input_dim, dtype=torch.float32)
    labels = torch.randn(data_size, output_dim, dtype=torch.float32)

    dataset = TensorDataset(input_data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    model = SimpleNet(input_dim, hidden_dim, output_dim)
    
    # 移除mixed_precision参数，使用配置文件中的设置（mixed_precision="no"）
    deepspeed = DeepSpeedPlugin(zero_stage=2, gradient_clipping=1)
    accelerator = Accelerator(deepspeed_plugin=deepspeed)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    
    # 使用Accelerator准备所有组件
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    for epoch in range(1000):
        model.train()
        total_train_loss = 0.0
        
        for batch in dataloader:
            inputs, labels = batch
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 使用accelerator进行反向传播
            accelerator.backward(loss)
            optimizer.step()
            
            # 累加训练损失
            total_train_loss += loss.item()
        
        # 评估阶段
        model.eval()
        total_eval_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                # 评估阶段不需要反向传播！
                total_eval_loss += loss.item()
        
        # 使用accelerator.print确保只在主进程输出
        avg_train_loss = total_train_loss / len(dataloader)
        avg_eval_loss = total_eval_loss / len(dataloader)
        
        accelerator.print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}")