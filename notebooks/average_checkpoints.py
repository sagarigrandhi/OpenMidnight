import torch
from pathlib import Path
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Average checkpoint weights')
parser.add_argument('--ckpt_dir', type=str, required=True, help='Directory containing training checkpoints')
parser.add_argument('--start', type=int, required=True, help='Start step for averaging')
parser.add_argument('--end', type=int, required=True, help='End step for averaging')
parser.add_argument('--step', type=int, required=True, help='Step interval')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory for averaged checkpoint')
args = parser.parse_args()

# For ablation evaluations, we average 5 checkpoints:
# --start 87500 --end 137500 --step 12500
# This gives checkpoints: 87500, 100000, 112500, 125000, 137500

# Generate steps
steps = list(range(args.start, args.end + 1, args.step))
print(f"Checkpoints to average: {steps}")
print(f"Number of checkpoints: {len(steps)}")

# Get checkpoint paths
ckpt_dir = Path(args.ckpt_dir)
ckpt_paths = [ckpt_dir / f"training_{step}" / "teacher_checkpoint.pth" for step in steps]

# Load checkpoints
state_dicts = [torch.load(str(p), map_location='cpu') for p in ckpt_paths if p.exists()]
print(f"Loaded {len(state_dicts)} checkpoints")
teacher_dicts = [sd["teacher"] for sd in state_dicts]

# Average weights
averaged_state_dict = {}
for key in teacher_dicts[0].keys():
    averaged_state_dict[key] = sum(td[key] for td in teacher_dicts) / len(teacher_dicts)

# Save
output_dict = {'teacher': averaged_state_dict}
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
torch.save(output_dict, output_dir / "teacher_checkpoint.pth")
print(f"Saved averaged checkpoint to: {output_dir / 'teacher_checkpoint.pth'}")