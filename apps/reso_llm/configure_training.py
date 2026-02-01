#!/usr/bin/env python3
"""
Interactive Training Configuration CLI for Reso-LLM.

Provides a menu-driven interface to:
- Select model size configuration
- Choose datasets (single or multiple)
- Configure training hyperparameters
- Save configuration to a JSON file for reproducible training

Usage:
    python configure_training.py
    python configure_training.py --config my_training.json  # Load existing config
    
    # Then train with:
    python train.py --from-config training_config.json
"""
import os
import sys
import json
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any

# Try to import curses for cross-platform support
try:
    import curses
    HAS_CURSES = True
except ImportError:
    HAS_CURSES = False


# Available datasets with descriptions
AVAILABLE_DATASETS = {
    "timdettmers/openassistant-guanaco": {
        "description": "High-quality conversations (10K examples)",
        "category": "conversation",
        "recommended": True
    },
    "databricks/databricks-dolly-15k": {
        "description": "Instruction following (15K examples)",
        "category": "instruction",
        "recommended": True
    },
    "tatsu-lab/alpaca": {
        "description": "Instruction tuning (52K examples)",
        "category": "instruction",
        "recommended": True
    },
    "Open-Orca/OpenOrca": {
        "description": "Large instruction dataset (1M+ examples)",
        "category": "instruction",
        "recommended": False
    },
    "WizardLM/WizardLM_evol_instruct_70k": {
        "description": "Evolved instructions (70K examples)",
        "category": "instruction",
        "recommended": False
    },
    "Anthropic/hh-rlhf": {
        "description": "Human preference data (170K examples)",
        "category": "conversation",
        "recommended": False
    },
    "OpenAssistant/oasst2": {
        "description": "Conversational assistant (90K+ examples)",
        "category": "conversation",
        "recommended": False
    },
    "lmsys/chatbot_arena_conversations": {
        "description": "Chatbot comparison data (33K examples)",
        "category": "conversation",
        "recommended": False
    },
    "sahil2801/CodeAlpaca-20k": {
        "description": "Code instructions (20K examples)",
        "category": "code",
        "recommended": False
    },
    "TokenBender/code_instructions_122k_alpaca_style": {
        "description": "Code instructions (122K examples)",
        "category": "code",
        "recommended": False
    },
    "lonestar108/sexygpt": {
        "description": "User/assistant pairs",
        "category": "conversation",
        "recommended": False
    },
    "lonestar108/enlightenedllm": {
        "description": "Instruction/output pairs",
        "category": "instruction",
        "recommended": False
    },
    "lonestar108/rawdata": {
        "description": "Input/output pairs",
        "category": "instruction",
        "recommended": False
    },
    "knkarthick/dialogsum": {
        "description": "Dialogue summarization (13K examples)",
        "category": "summarization",
        "recommended": False
    }
}

# Model size configurations
MODEL_SIZES = {
    "tiny": {
        "description": "4.5M params - Very fast training, testing",
        "dim": 256,
        "layers": 4,
        "heads": 4
    },
    "small": {
        "description": "26M params - Quick experiments",
        "dim": 512,
        "layers": 6,
        "heads": 8
    },
    "medium": {
        "description": "117M params - Good balance (default)",
        "dim": 768,
        "layers": 12,
        "heads": 12
    },
    "large": {
        "description": "345M params - High quality",
        "dim": 1024,
        "layers": 24,
        "heads": 16
    },
    "xl": {
        "description": "1.5B params - Cutting-edge performance",
        "dim": 1600,
        "layers": 48,
        "heads": 25
    }
}


@dataclass
class TrainingConfig:
    """Training configuration that can be saved/loaded."""
    # Model
    model_size: str = "medium"
    
    # Datasets
    datasets: List[str] = field(default_factory=lambda: ["timdettmers/openassistant-guanaco"])
    max_tokens_per_dataset: int = 5_000_000
    
    # Training hyperparameters
    epochs: int = 5
    batch_size: int = 32
    seq_len: int = 256
    learning_rate: float = 3e-4
    warmup_steps: int = 100
    
    # Options
    shuffle_data: bool = True
    validate_output: bool = True
    reset_checkpoint: bool = False
    
    # Extensions (Nonstandard)
    use_agency: bool = False
    use_prsc: bool = False
    use_temporal_smf: bool = False
    use_entanglement: bool = False
    use_stability: bool = False
    use_stochastic_resonance: bool = False

    # Metadata
    config_name: str = "default"
    description: str = ""
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        print(f"\n✓ Configuration saved to: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def to_cli_args(self) -> str:
        """Convert to CLI arguments string."""
        # Note: Extended features are only supported via JSON config loading
        # so we don't add them to CLI args here.
        args = [
            f"--config {self.model_size}",
            f"--epochs {self.epochs}",
            f"--batch-size {self.batch_size}",
            f"--seq-len {self.seq_len}",
            f"--lr {self.learning_rate}",
            f"--warmup {self.warmup_steps}",
        ]
        
        if len(self.datasets) > 1:
            args.append(f"--datasets {' '.join(self.datasets)}")
            args.append(f"--max-tokens-per-dataset {self.max_tokens_per_dataset}")
        else:
            args.append(f"--dataset {self.datasets[0]}")
        
        if self.reset_checkpoint:
            args.append("--reset")
        
        return " ".join(args)


class SimpleMenu:
    """Simple terminal menu without external dependencies."""
    
    def __init__(self, title: str):
        self.title = title
        self.items: List[str] = []
        self.selected = 0
        
    def add_item(self, text: str):
        """Add an item to the menu."""
        self.items.append(text)
    
    def show(self) -> int:
        """Show menu and return selected index."""
        if HAS_CURSES:
            return self._show_curses()
        else:
            return self._show_simple()
    
    def _show_simple(self) -> int:
        """Simple numbered menu for systems without curses."""
        print(f"\n{'='*60}")
        print(f"  {self.title}")
        print('='*60)
        
        for i, item in enumerate(self.items):
            marker = ">" if i == 0 else " "
            print(f"  {i+1}. {item}")
        
        print()
        while True:
            try:
                choice = input("Enter number (or 'q' to quit): ").strip()
                if choice.lower() == 'q':
                    return -1
                idx = int(choice) - 1
                if 0 <= idx < len(self.items):
                    return idx
                print(f"Please enter 1-{len(self.items)}")
            except ValueError:
                print("Please enter a valid number")
    
    def _show_curses(self) -> int:
        """Interactive curses-based menu."""
        def draw_menu(stdscr):
            curses.curs_set(0)  # Hide cursor
            current = 0
            
            while True:
                stdscr.clear()
                h, w = stdscr.getmaxyx()
                
                # Title
                title_x = max(0, (w - len(self.title)) // 2)
                stdscr.addstr(1, title_x, self.title, curses.A_BOLD)
                stdscr.addstr(2, 0, "=" * min(w-1, 60))
                
                # Items
                for i, item in enumerate(self.items):
                    y = 4 + i
                    if y >= h - 2:
                        break
                    
                    if i == current:
                        stdscr.addstr(y, 2, f"> {item}", curses.A_REVERSE)
                    else:
                        stdscr.addstr(y, 2, f"  {item}")
                
                # Help
                help_text = "↑/↓: Navigate | Enter: Select | q: Quit"
                stdscr.addstr(h-1, 0, help_text[:w-1])
                
                stdscr.refresh()
                
                key = stdscr.getch()
                
                if key == curses.KEY_UP and current > 0:
                    current -= 1
                elif key == curses.KEY_DOWN and current < len(self.items) - 1:
                    current += 1
                elif key == ord('\n'):
                    return current
                elif key == ord('q'):
                    return -1
        
        try:
            return curses.wrapper(draw_menu)
        except Exception:
            return self._show_simple()


class MultiSelectMenu:
    """Menu allowing multiple item selection."""
    
    def __init__(self, title: str):
        self.title = title
        self.items: List[str] = []
        self.selected: List[bool] = []
        self.descriptions: List[str] = []
    
    def add_item(self, text: str, description: str = "", preselected: bool = False):
        """Add an item to the menu."""
        self.items.append(text)
        self.descriptions.append(description)
        self.selected.append(preselected)
    
    def show(self) -> List[int]:
        """Show menu and return list of selected indices."""
        if HAS_CURSES:
            return self._show_curses()
        else:
            return self._show_simple()
    
    def _show_simple(self) -> List[int]:
        """Simple text-based multi-select."""
        # Store original selections in case of cancel
        original_selected = self.selected.copy()
        
        print(f"\n{'='*60}")
        print(f"  {self.title}")
        print('='*60)
        print("Enter numbers to toggle (comma-separated), 'done' to confirm, 'q' to cancel")
        print()
        
        while True:
            for i, (item, desc) in enumerate(zip(self.items, self.descriptions)):
                check = "[x]" if self.selected[i] else "[ ]"
                if desc:
                    print(f"  {i+1}. {check} {item}")
                    print(f"       {desc}")
                else:
                    print(f"  {i+1}. {check} {item}")
            
            choice = input("\nToggle (e.g., 1,2,3) or 'done'/'q': ").strip()
            
            if choice.lower() == 'done':
                break
            elif choice.lower() == 'q':
                # Restore original selections and return them (cancel)
                self.selected = original_selected
                return [i for i, sel in enumerate(self.selected) if sel]
            
            try:
                indices = [int(x.strip()) - 1 for x in choice.split(',')]
                for idx in indices:
                    if 0 <= idx < len(self.items):
                        self.selected[idx] = not self.selected[idx]
            except ValueError:
                print("Please enter valid numbers")
            
            print()  # Clear for redraw
        
        return [i for i, sel in enumerate(self.selected) if sel]
    
    def _show_curses(self) -> List[int]:
        """Curses-based multi-select menu."""
        # Store original selections in case of cancel
        original_selected = self.selected.copy()
        
        def draw_menu(stdscr):
            curses.curs_set(0)
            current = 0
            
            while True:
                stdscr.clear()
                h, w = stdscr.getmaxyx()
                
                # Title
                title_x = max(0, (w - len(self.title)) // 2)
                stdscr.addstr(1, title_x, self.title, curses.A_BOLD)
                stdscr.addstr(2, 0, "=" * min(w-1, 60))
                
                # Items
                y = 4
                for i, (item, desc) in enumerate(zip(self.items, self.descriptions)):
                    if y >= h - 3:
                        break
                    
                    check = "[x]" if self.selected[i] else "[ ]"
                    text = f"{check} {item}"
                    
                    if i == current:
                        stdscr.addstr(y, 2, text[:w-3], curses.A_REVERSE)
                    else:
                        stdscr.addstr(y, 2, text[:w-3])
                    
                    if desc:
                        y += 1
                        if y < h - 3:
                            stdscr.addstr(y, 6, desc[:w-7], curses.A_DIM)
                    y += 1
                
                # Help
                help_text = "↑/↓: Navigate | Space: Toggle | Enter: Done | q: Cancel"
                stdscr.addstr(h-1, 0, help_text[:w-1])
                
                # Count
                count = sum(self.selected)
                stdscr.addstr(h-2, 2, f"Selected: {count} item(s)")
                
                stdscr.refresh()
                
                key = stdscr.getch()
                
                if key == curses.KEY_UP and current > 0:
                    current -= 1
                elif key == curses.KEY_DOWN and current < len(self.items) - 1:
                    current += 1
                elif key == ord(' '):
                    self.selected[current] = not self.selected[current]
                elif key == ord('\n'):
                    return [i for i, sel in enumerate(self.selected) if sel]
                elif key == ord('q'):
                    # Restore original and return (cancel)
                    self.selected = original_selected.copy()
                    return [i for i, sel in enumerate(self.selected) if sel]
        
        try:
            return curses.wrapper(draw_menu)
        except Exception:
            return self._show_simple()


class NumberInput:
    """Input for numeric values with validation."""
    
    def __init__(self, prompt: str, default: Any, min_val: Any = None, max_val: Any = None):
        self.prompt = prompt
        self.default = default
        self.min_val = min_val
        self.max_val = max_val
    
    def show(self) -> Any:
        """Show input and return value."""
        print(f"\n{self.prompt}")
        print(f"  Default: {self.default}")
        
        if self.min_val is not None and self.max_val is not None:
            print(f"  Range: {self.min_val} - {self.max_val}")
        
        while True:
            value = input("Enter value (or press Enter for default): ").strip()
            
            if not value:
                return self.default
            
            try:
                # Handle float or int
                if isinstance(self.default, float):
                    val = float(value)
                else:
                    val = int(value)
                
                if self.min_val is not None and val < self.min_val:
                    print(f"Value must be >= {self.min_val}")
                    continue
                if self.max_val is not None and val > self.max_val:
                    print(f"Value must be <= {self.max_val}")
                    continue
                
                return val
            except ValueError:
                print("Please enter a valid number")


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print the configuration tool header."""
    print("=" * 60)
    print("   Reso-LLM Training Configuration")
    print("   Interactive setup for model training")
    print("=" * 60)


def print_config_summary(config: TrainingConfig):
    """Print a summary of the current configuration."""
    print("\n" + "=" * 60)
    print("  Current Configuration")
    print("=" * 60)
    print(f"\n  Model Size: {config.model_size}")
    size_info = MODEL_SIZES.get(config.model_size, {})
    if size_info:
        print(f"    → {size_info.get('description', '')}")
    
    print(f"\n  Datasets ({len(config.datasets)}):")
    for ds in config.datasets:
        info = AVAILABLE_DATASETS.get(ds, {})
        desc = info.get('description', '')
        print(f"    • {ds}")
        if desc:
            print(f"      {desc}")
    
    print(f"\n  Training Parameters:")
    print(f"    Epochs:         {config.epochs}")
    print(f"    Batch Size:     {config.batch_size}")
    print(f"    Sequence Length:{config.seq_len}")
    print(f"    Learning Rate:  {config.learning_rate}")
    print(f"    Warmup Steps:   {config.warmup_steps}")
    
    print(f"\n  Options:")
    print(f"    Shuffle Data:   {'Yes' if config.shuffle_data else 'No'}")
    print(f"    Validate Output:{'Yes' if config.validate_output else 'No'}")
    print(f"    Reset Checkpoint:{'Yes' if config.reset_checkpoint else 'No'}")
    
    if config.max_tokens_per_dataset != 5_000_000:
        print(f"    Max Tokens/DS:  {config.max_tokens_per_dataset:,}")
    
    print(f"\n  Extensions (Nonstandard):")
    print(f"    Agency:         {'Enabled' if config.use_agency else 'Disabled'}")
    print(f"    PRSC:           {'Enabled' if config.use_prsc else 'Disabled'}")
    print(f"    Temporal SMF:   {'Enabled' if config.use_temporal_smf else 'Disabled'}")
    print(f"    Entanglement:   {'Enabled' if config.use_entanglement else 'Disabled'}")
    print(f"    Stability:      {'Enabled' if config.use_stability else 'Disabled'}")
    print(f"    Stoch. Res.:    {'Enabled' if config.use_stochastic_resonance else 'Disabled'}")

    print("=" * 60)


def configure_model(config: TrainingConfig) -> TrainingConfig:
    """Configure model size."""
    menu = SimpleMenu("Select Model Size")
    
    sizes = list(MODEL_SIZES.keys())
    for size in sizes:
        info = MODEL_SIZES[size]
        menu.add_item(f"{size.upper()} - {info['description']}")
    
    idx = menu.show()
    if idx >= 0:
        config.model_size = sizes[idx]
    
    return config


def configure_datasets(config: TrainingConfig) -> TrainingConfig:
    """Configure datasets to use."""
    menu = MultiSelectMenu("Select Datasets for Training")
    
    datasets = list(AVAILABLE_DATASETS.keys())
    for ds in datasets:
        info = AVAILABLE_DATASETS[ds]
        preselected = ds in config.datasets
        rec = " ⭐ RECOMMENDED" if info.get('recommended') else ""
        menu.add_item(
            f"{ds}{rec}",
            f"[{info['category']}] {info['description']}",
            preselected=preselected
        )
    
    selected = menu.show()
    
    if selected:
        config.datasets = [datasets[i] for i in selected]
    else:
        # Keep at least one dataset
        print("\n⚠ No datasets selected, keeping current selection.")
    
    return config


def configure_extensions(config: TrainingConfig) -> TrainingConfig:
    """Configure optional nonstandard extensions."""
    menu = MultiSelectMenu("Configure Extensions (Nonstandard)")
    
    items = [
        ("Agency Layer", "Self-directed attention and goal formation", config.use_agency),
        ("PRSC Semantic Layer", "Compositional semantics via prime interference", config.use_prsc),
        ("Temporal SMF", "Holographic memory with episodic tagging", config.use_temporal_smf),
        ("Entanglement Network", "Multi-agent coordination capabilities", config.use_entanglement),
        ("Stability Monitor", "Predictive Lyapunov analysis for chaos detection", config.use_stability),
        ("Stochastic Resonance", "Controlled noise injection to escape local minima", config.use_stochastic_resonance),
    ]
    
    for label, desc, selected in items:
        menu.add_item(label, desc, preselected=selected)
        
    selected_indices = menu.show()
    
    # If user cancelled (pressed 'q'), keep current config
    if selected_indices is None or (isinstance(selected_indices, list) and len(selected_indices) == 0 and not any(menu.selected)):
        # Check if user explicitly unselected everything vs cancelled
        # If any were pre-selected and now none are, that's a cancel
        had_preselections = any([config.use_agency, config.use_prsc, config.use_temporal_smf,
                                  config.use_entanglement, config.use_stability, config.use_stochastic_resonance])
        if had_preselections and len(selected_indices) == 0:
            print("\n⚠ Selection cancelled, keeping current extensions.")
            return config
    
    # Update based on selection
    config.use_agency = 0 in selected_indices
    config.use_prsc = 1 in selected_indices
    config.use_temporal_smf = 2 in selected_indices
    config.use_entanglement = 3 in selected_indices
    config.use_stability = 4 in selected_indices
    config.use_stochastic_resonance = 5 in selected_indices
        
    return config


def configure_hyperparameters(config: TrainingConfig) -> TrainingConfig:
    """Configure training hyperparameters."""
    print("\n" + "=" * 60)
    print("  Training Hyperparameters")
    print("=" * 60)
    
    config.epochs = NumberInput(
        "Number of Epochs:",
        default=config.epochs,
        min_val=1,
        max_val=100
    ).show()
    
    config.batch_size = NumberInput(
        "Batch Size:",
        default=config.batch_size,
        min_val=1,
        max_val=256
    ).show()
    
    config.seq_len = NumberInput(
        "Sequence Length:",
        default=config.seq_len,
        min_val=64,
        max_val=2048
    ).show()
    
    config.learning_rate = NumberInput(
        "Learning Rate:",
        default=config.learning_rate,
        min_val=1e-6,
        max_val=1e-2
    ).show()
    
    config.warmup_steps = NumberInput(
        "Warmup Steps:",
        default=config.warmup_steps,
        min_val=0,
        max_val=10000
    ).show()
    
    if len(config.datasets) > 1:
        config.max_tokens_per_dataset = NumberInput(
            "Max Tokens per Dataset:",
            default=config.max_tokens_per_dataset,
            min_val=100_000,
            max_val=50_000_000
        ).show()
    
    return config


def configure_options(config: TrainingConfig) -> TrainingConfig:
    """Configure training options."""
    menu = SimpleMenu("Toggle Training Options")
    
    menu.add_item(f"Shuffle Data: {'ON' if config.shuffle_data else 'OFF'}")
    menu.add_item(f"Validate Output: {'ON' if config.validate_output else 'OFF'}")
    menu.add_item(f"Reset Checkpoint: {'ON' if config.reset_checkpoint else 'OFF'}")
    menu.add_item("← Back to Main Menu")
    
    idx = menu.show()
    
    if idx == 0:
        config.shuffle_data = not config.shuffle_data
        return configure_options(config)
    elif idx == 1:
        config.validate_output = not config.validate_output
        return configure_options(config)
    elif idx == 2:
        config.reset_checkpoint = not config.reset_checkpoint
        return configure_options(config)
    
    return config


def use_preset(config: TrainingConfig) -> TrainingConfig:
    """Apply a preset configuration."""
    menu = SimpleMenu("Select Preset Configuration")
    
    menu.add_item("Quick Test - tiny model, small dataset, 1 epoch")
    menu.add_item("Development - small model, 1 dataset, 3 epochs")
    menu.add_item("Balanced - medium model, 3 datasets, 5 epochs (recommended)")
    menu.add_item("Quality - large model, 3 datasets, 10 epochs")
    menu.add_item("Maximum - XL model, all datasets, 20 epochs")
    menu.add_item("← Cancel")
    
    idx = menu.show()
    
    if idx == 0:  # Quick Test
        config.model_size = "tiny"
        config.datasets = ["timdettmers/openassistant-guanaco"]
        config.epochs = 1
        config.batch_size = 16
        config.max_tokens_per_dataset = 500_000
    elif idx == 1:  # Development
        config.model_size = "small"
        config.datasets = ["timdettmers/openassistant-guanaco"]
        config.epochs = 3
        config.batch_size = 32
    elif idx == 2:  # Balanced
        config.model_size = "medium"
        config.datasets = [
            "timdettmers/openassistant-guanaco",
            "databricks/databricks-dolly-15k",
            "tatsu-lab/alpaca"
        ]
        config.epochs = 5
        config.batch_size = 32
        config.max_tokens_per_dataset = 2_000_000
    elif idx == 3:  # Quality
        config.model_size = "large"
        config.datasets = [
            "timdettmers/openassistant-guanaco",
            "databricks/databricks-dolly-15k",
            "tatsu-lab/alpaca"
        ]
        config.epochs = 10
        config.batch_size = 16
        config.max_tokens_per_dataset = 5_000_000
    elif idx == 4:  # Maximum
        config.model_size = "xl"
        config.datasets = [ds for ds in AVAILABLE_DATASETS.keys()][:8]  # Top 8
        config.epochs = 20
        config.batch_size = 8
        config.max_tokens_per_dataset = 3_000_000
    
    return config


def save_config(config: TrainingConfig):
    """Save configuration to file."""
    print("\n" + "=" * 60)
    print("  Save Configuration")
    print("=" * 60)
    
    # Get config name
    name = input(f"Configuration name [{config.config_name}]: ").strip()
    if name:
        config.config_name = name
    
    # Get description
    desc = input("Description (optional): ").strip()
    config.description = desc
    
    # Default filename
    default_filename = f"training_config_{config.config_name}.json"
    filename = input(f"Filename [{default_filename}]: ").strip()
    if not filename:
        filename = default_filename
    
    # Ensure .json extension
    if not filename.endswith('.json'):
        filename += '.json'
    
    # Save to apps/reso_llm directory
    save_path = os.path.join(os.path.dirname(__file__), filename)
    config.save(save_path)
    
    # Also show CLI command
    print("\nTo train with this configuration, run:")
    print(f"  python train.py --from-config {save_path}")
    print("\nOr manually:")
    print(f"  python train.py {config.to_cli_args()}")


def main_menu(config: TrainingConfig) -> TrainingConfig:
    """Show main menu and handle selection."""
    menu = SimpleMenu("Main Menu")
    
    menu.add_item("📦 Select Model Size")
    menu.add_item("📚 Select Datasets")
    menu.add_item("⚙️  Configure Hyperparameters")
    menu.add_item("🧩 Configure Extensions")
    menu.add_item("🔧 Toggle Options")
    menu.add_item("📋 Use Preset Configuration")
    menu.add_item("📄 View Current Configuration")
    menu.add_item("💾 Save Configuration")
    menu.add_item("🚀 Save and Start Training")
    menu.add_item("❌ Exit")
    
    idx = menu.show()
    
    if idx == 0:
        config = configure_model(config)
    elif idx == 1:
        config = configure_datasets(config)
    elif idx == 2:
        config = configure_hyperparameters(config)
    elif idx == 3:
        config = configure_extensions(config)
    elif idx == 4:
        config = configure_options(config)
    elif idx == 5:
        config = use_preset(config)
    elif idx == 6:
        print_config_summary(config)
        input("\nPress Enter to continue...")
    elif idx == 7:
        save_config(config)
        input("\nPress Enter to continue...")
    elif idx == 8:
        # Save and run
        save_config(config)
        print("\nStarting training...")
        
        # Build command
        # We must use --from-config to support extensions
        # We need to find the filename we just saved to
        default_filename = f"training_config_{config.config_name}.json"
        save_path = os.path.join(os.path.dirname(__file__), default_filename)
        
        cmd = f"python {os.path.join(os.path.dirname(__file__), 'train.py')} --from-config {save_path}"
        print(f"\nExecuting: {cmd}\n")
        os.system(cmd)
        return config
    elif idx == 9 or idx == -1:
        print("\nExiting configuration tool.")
        sys.exit(0)
    
    return config


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Configure Reso-LLM Training")
    parser.add_argument("--config", type=str, default=None,
                       help="Load existing configuration file")
    args = parser.parse_args()
    
    # Initialize or load config
    if args.config and os.path.exists(args.config):
        print(f"Loading configuration from: {args.config}")
        config = TrainingConfig.load(args.config)
    else:
        config = TrainingConfig()
    
    # Main loop
    while True:
        clear_screen()
        print_header()
        print_config_summary(config)
        config = main_menu(config)


if __name__ == "__main__":
    main()
