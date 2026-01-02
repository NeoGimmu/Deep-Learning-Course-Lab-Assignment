import subprocess
import json
import os
import time
import argparse
import sys

# 定义要测试的模型配置
model_configurations = [
    {
        'name': 'Original Transformer',
        'model_type': 'transformer',
        'pos_encoding': 'original',
        'tokenizer': 'bpe',
        'description': 'Transformer with original positional encoding and BPE tokenization'
    },
    {
        'name': 'Transformer with RoPE',
        'model_type': 'transformer',
        'pos_encoding': 'rope',
        'tokenizer': 'bpe',
        'description': 'Transformer with rotary positional encoding and BPE tokenization'
    },
    {
        'name': 'Transformer with SentencePiece',
        'model_type': 'transformer',
        'pos_encoding': 'original',
        'tokenizer': 'sentencepiece',
        'description': 'Transformer with original positional encoding and SentencePiece tokenization'
    },
    {
        'name': 'Mamba Transformer',
        'model_type': 'mamba',
        'pos_encoding': 'original',  # Mamba不需要位置编码，但需要传递有效值
        'tokenizer': 'bpe',
        'description': 'Mamba model with BPE tokenization'
    }
]

# 命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='Comparison Experiments for Transformer Variants and Mamba')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to use for all experiments')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for all experiments')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs for all experiments')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate for all experiments')
    parser.add_argument('--d_model', type=int, default=512,
                        help='Model dimension for all experiments')
    parser.add_argument('--results_file', type=str, default='experiment_results.json',
                        help='File to save experiment results')
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip dataset download if already downloaded')
    return parser.parse_args()

# 运行单个实验
def run_experiment(config, args):
    print(f"\n{'='*60}")
    print(f"Running experiment: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"{'='*60}")
    
    # 构建命令
    cmd = [
        sys.executable, 'training_and_testing.py',
        '--model_type', config['model_type'],
        '--pos_encoding', config['pos_encoding'],
        '--tokenizer', config['tokenizer'],
        '--device', args.device,
        '--batch_size', str(args.batch_size),
        '--epochs', str(args.epochs),
        '--learning_rate', str(args.learning_rate),
        '--d_model', str(args.d_model)
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    # 运行命令
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        experiment_time = time.time() - start_time
        
        # 检查是否成功
        if result.returncode == 0:
            print(f"Experiment completed successfully in {experiment_time:.2f}s")
            print(f"STDOUT: {result.stdout[:500]}...")  # 只显示前500个字符
            return {
                'config': config,
                'success': True,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'experiment_time': experiment_time
            }
        else:
            print(f"Experiment failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return {
                'config': config,
                'success': False,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'experiment_time': experiment_time
            }
    except Exception as e:
        print(f"Experiment failed with exception: {e}")
        return {
            'config': config,
            'success': False,
            'stdout': '',
            'stderr': str(e),
            'experiment_time': time.time() - start_time
        }

# 下载数据集
def download_dataset(skip_download):
    if skip_download:
        print("Skipping dataset download...")
        return True
    
    print("\n" + "="*60)
    print("Downloading and preprocessing dataset...")
    print("="*60)
    
    cmd = [sys.executable, 'download_dataset.py']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            print("Dataset downloaded and preprocessed successfully")
            return True
        else:
            print(f"Dataset download failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return False
    except Exception as e:
        print(f"Dataset download failed with exception: {e}")
        return False

# 解析训练结果
def parse_training_results(results):
    parsed_results = []
    
    for result in results:
        if not result['success']:
            parsed_results.append({
                'name': result['config']['name'],
                'success': False,
                'total_training_time': float('inf'),
                'test_loss': float('inf'),
                'error': result['stderr']
            })
            continue
        
        # 从stdout中提取测试损失
        stdout = result['stdout']
        test_loss = None
        total_training_time = result['experiment_time']
        
        for line in stdout.split('\n'):
            if 'Test Loss:' in line:
                try:
                    test_loss = float(line.split(':')[1].strip())
                except ValueError:
                    pass
        
        # 如果没有找到测试损失，尝试从结果文件中读取
        if test_loss is None:
            checkpoint_dir = 'checkpoints'
            model_type = result['config']['model_type']
            pos_encoding = result['config']['pos_encoding']
            tokenizer = result['config']['tokenizer']
            results_file = os.path.join(checkpoint_dir, f'training_results_{model_type}_{pos_encoding}_{tokenizer}.json')
            
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r') as f:
                        training_results = json.load(f)
                        test_loss = training_results.get('test_loss', None)
                        if 'total_training_time' in training_results:
                            total_training_time = training_results['total_training_time']
                except json.JSONDecodeError:
                    pass
        
        parsed_results.append({
            'name': result['config']['name'],
            'success': True,
            'model_type': result['config']['model_type'],
            'pos_encoding': result['config']['pos_encoding'],
            'tokenizer': result['config']['tokenizer'],
            'total_training_time': total_training_time,
            'test_loss': test_loss if test_loss is not None else float('inf'),
            'description': result['config']['description']
        })
    
    return parsed_results

# 生成实验报告
def generate_report(parsed_results, args):
    print(f"\n{'='*80}")
    print("EXPERIMENT REPORT")
    print(f"{'='*80}")
    print(f"Experiment Parameters:")
    print(f"  Device: {args.device}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Model Dimension: {args.d_model}")
    print(f"{'='*80}")
    
    print("\nResults Summary:")
    print(f"{'Model Name':<30} {'Training Time (s)':<20} {'Test Loss':<15} {'Status':<10}")
    print(f"{'-'*75}")
    
    for result in parsed_results:
        status = "SUCCESS" if result['success'] else "FAILED"
        time_str = f"{result['total_training_time']:.2f}" if result['success'] else "N/A"
        loss_str = f"{result['test_loss']:.4f}" if result['success'] and result['test_loss'] < float('inf') else "N/A"
        print(f"{result['name']:<30} {time_str:<20} {loss_str:<15} {status:<10}")
    
    print(f"{'='*80}")
    
    # 分析结果
    print("\nAnalysis:")
    
    # 找出训练最快的模型
    successful_results = [r for r in parsed_results if r['success']]
    if successful_results:
        fastest_model = min(successful_results, key=lambda x: x['total_training_time'])
        print(f"\n1. Fastest Training: {fastest_model['name']}")
        print(f"   Training Time: {fastest_model['total_training_time']:.2f}s")
        
        # 找出测试损失最小的模型
        best_model = min(successful_results, key=lambda x: x['test_loss'])
        print(f"\n2. Best Performance (Lowest Test Loss): {best_model['name']}")
        print(f"   Test Loss: {best_model['test_loss']:.4f}")
        
        # 比较不同位置编码
        transformer_results = [r for r in successful_results if r['model_type'] == 'transformer']
        if transformer_results:
            print(f"\n3. Transformer Variants Comparison:")
            for result in transformer_results:
                print(f"   - {result['name']}: Test Loss = {result['test_loss']:.4f}, Training Time = {result['total_training_time']:.2f}s")
        
        # 比较Mamba和Transformer
        mamba_results = [r for r in successful_results if r['model_type'] == 'mamba']
        if mamba_results and transformer_results:
            print(f"\n4. Mamba vs Transformer:")
            mamba_result = mamba_results[0]
            avg_transformer_time = sum(r['total_training_time'] for r in transformer_results) / len(transformer_results)
            avg_transformer_loss = sum(r['test_loss'] for r in transformer_results) / len(transformer_results)
            print(f"   - Mamba: Test Loss = {mamba_result['test_loss']:.4f}, Training Time = {mamba_result['total_training_time']:.2f}s")
            print(f"   - Avg Transformer: Test Loss = {avg_transformer_loss:.4f}, Training Time = {avg_transformer_time:.2f}s")
    
    print(f"\n{'='*80}")

# 主函数
def main():
    args = parse_args()
    
    # 创建检查点目录
    os.makedirs('checkpoints', exist_ok=True)
    
    # 下载数据集
    if not args.skip_download:
        download_success = download_dataset(args.skip_download)
        if not download_success:
            print("Dataset download failed. Exiting...")
            return
    
    # 运行所有实验
    results = []
    for config in model_configurations:
        result = run_experiment(config, args)
        results.append(result)
    
    # 解析结果
    parsed_results = parse_training_results(results)
    
    # 保存结果
    with open(args.results_file, 'w') as f:
        json.dump(parsed_results, f, indent=4)
    
    print(f"\nAll experiments completed! Results saved to {args.results_file}")
    
    # 生成报告
    generate_report(parsed_results, args)

if __name__ == '__main__':
    main()
