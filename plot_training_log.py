import csv
import matplotlib.pyplot as plt
import os

def plot_logs(csv_path, output_dir):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # Read CSV
    epochs = []
    train_loss = []
    val_loss = []
    train_components = {}
    val_components = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        # Initialize component lists
        loss_component_names = [f for f in fieldnames if (f.startswith('Train_loss_') or f.startswith('Val_loss_')) and f not in ['Train_Loss', 'Val_Loss']]
        for name in loss_component_names:
            if name.startswith('Train_loss_'):
                if 'train' not in train_components: train_components = {} # logic simplification
                train_components[name] = []
            else:
                if 'val' not in val_components: val_components = {}
                val_components[name] = []

        # Re-initialize explicitly for clarity
        train_comp_data = {name: [] for name in fieldnames if name.startswith('Train_loss_') and name != 'Train_Loss'}
        val_comp_data = {name: [] for name in fieldnames if name.startswith('Val_loss_') and name != 'Val_Loss'}

        for row in reader:
            epochs.append(float(row['Epoch']))
            train_loss.append(float(row['Train_Loss']))
            val_loss.append(float(row['Val_Loss']))
            
            for name in train_comp_data:
                train_comp_data[name].append(float(row[name]))
            for name in val_comp_data:
                val_comp_data[name].append(float(row[name]))

    # Plot Total Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()

    # Plot Components
    if train_comp_data:
        plt.figure(figsize=(12, 8))
        for name, values in train_comp_data.items():
            plt.plot(epochs, values, label=name, linestyle='-')
        for name, values in val_comp_data.items():
            plt.plot(epochs, values, label=name, linestyle='--')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss Component')
        plt.title('Loss Components Breakdown')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'loss_components_plot.png'))
        plt.close()

    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    base_dir = "/home/takeshi/Documents/AOL DL"
    csv_file = os.path.join(base_dir, "checkpoints_5/training_log.csv")
    output_folder = os.path.join(base_dir, "checkpoints_5")
    plot_logs(csv_file, output_folder)
