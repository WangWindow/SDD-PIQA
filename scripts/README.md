# Scripts / 脚本

This folder contains shell scripts to automate the training and data generation processes. All scripts are configured to run in the background and save logs to the `logs/` directory.
本文件夹包含用于自动化训练和数据生成流程的 Shell 脚本。所有脚本均配置为在后台运行，并将日志保存到 `logs/` 目录。

## Usage / 使用方法

Run the scripts from the project root or the `scripts/` directory.
从项目根目录或 `scripts/` 目录运行脚本。

```bash
bash scripts/script_name.sh
```

## Contents / 内容

*   **`gen_pseudo_labels.sh`**:
    *   **Pseudo Label Generation**: Runs the 3-step process to generate quality pseudo-labels.
    *   **伪标签生成**: 运行 3 步流程以生成质量伪标签。

*   **`kill_all.sh`**:
    *   **Stop Processes**: Kills all running background processes related to this project (training, generation, etc.).
    *   **停止进程**: 终止所有与本项目相关的后台运行进程（训练、生成等）。

*   **`train.sh`**:
    *   **Quality Model Training**: Trains the final quality regression model using the generated pseudo-labels.
    *   **质量模型训练**: 使用生成的伪标签训练最终的质量回归模型。

*   **`train_rec.sh`**:
    *   **Recognition Training**: Trains the ResNet50 backbone for palmprint recognition.
    *   **识别训练**: 训练用于掌纹识别的 ResNet50 主干网络。


*   **`run_all.sh`**:
    *   **Full Pipeline**: Sequentially runs the entire pipeline: Recognition Training -> Pseudo Label Generation -> Quality Model Training.
    *   **全流程**: 按顺序运行整个流程：识别训练 -> 伪标签生成 -> 质量模型训练。
