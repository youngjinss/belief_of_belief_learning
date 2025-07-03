from Data_Handlers import FDataLoader as DL
from Data_Handlers import FDataProcessor as DP

from ToMnetEngine.ToMnetF import ToMnet

import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

"""
@Author Filip Borowiak 
"""


HEIGHT = 13  # map size
WIDTH = 13  # map size
EPOCHS = 100
BATCH = 512
TS = 10  # Trajectory size / length = time frames
DEPTH = 10  # depth of the tensor (channels)
TRAINING_PROPORTION = 0.9
LR = 1e-04
EXPERIMENT_NO = 2

dir = "./data/Saved Games/Experiment2/"

dl = DL.DataReader(TS, HEIGHT, WIDTH, DEPTH, EXPERIMENT_NO)

all_games = dl.LoadAllGames(use_percentage=1.0, directory=dir)
print(f"Total size: {len(all_games)}")

dp = DP.DataProcessor(TS, HEIGHT, WIDTH, DEPTH)

all_games = dp.zeroPadding(TS, all_games)

data_trajectories = all_games["traj_history"]
data_trajectories_zp = all_games["traj_history_zp"]
data_current_state = all_games["current_state_history"]
data_actions = all_games["actions_history"]
data_labels = all_games["labels_history"]


data_traj = torch.tensor(
    data_trajectories_zp, dtype=torch.float32
)  # long charnet input
data_curr = torch.tensor(data_current_state, dtype=torch.float32)  # short pred input
data_act = torch.tensor(data_actions, dtype=torch.float32)
data_labels = torch.tensor(data_labels, dtype=torch.float32)

# data traj shape (Batch x Channel(Depth) x Width, Height x Time_frames/Trajctory_Size)

dataset = TensorDataset(data_traj, data_curr, data_act)


seed = torch.Generator().manual_seed(42)

total_size = len(dataset)
train_size = int(total_size * TRAINING_PROPORTION)

val_size = int(total_size - train_size)

trainDataSet = TensorDataset(
    data_traj[:train_size, ...], data_curr[:train_size, ...], data_act[:train_size, ...]
)
valDataSet = TensorDataset(
    data_traj[train_size:, ...], data_curr[train_size:, ...], data_act[train_size:, ...]
)
TrainLoader = DataLoader(trainDataSet, batch_size=BATCH, shuffle=False, drop_last=True)
ValLoader = DataLoader(valDataSet, batch_size=BATCH, shuffle=False, drop_last=True)
print("Trainining size: ", len(TrainLoader.dataset))
print("Validation size: ", len(ValLoader.dataset))


print("Training loop \n")


print("one-hot-encoding targets")


print("\n---T-R-A-I-N-I-N-G--T-O-M-N-E-T--N-E-U-R-A-L--N-E-T-W-O-R-K---\n")

# PAPER: batch_size: 16, lr 1e-4, minibatch: 40K, hence for 1210 tr samples, epoch = 529 (+-) 1 batch
print(
    "----------------------------------------------------------------------------------------"
)

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f"Moving to device: {device} \n")

model = ToMnet(
    Batch=BATCH,
    ResidualBlocks=5,
    N_echar=8,
    out_channels=32,
    Max_trajectory_size=TS,
    Width=WIDTH,
    Height=HEIGHT,
    Depth=DEPTH,
)
model = model.to(device)


pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model Parameters: {pytorch_total_params}")

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.001)


def train():

    train_accuracy = []
    train_loss = []

    val_accuracy = []
    val_loss = []

    for epoch in range(EPOCHS):
        running_loss_train = 0.0
        running_loss_val = 0.0

        all_pred_train = 0
        all_pred_val = 0

        correct_pred_train = 0
        correct_pred_val = 0

        model.train()
        for idx, data in enumerate(TrainLoader, 0):
            traj, curr, act = data
            traj, curr, act = traj.to(device), curr.to(device), act.to(device)
            act = act.squeeze()
            act = act.type(torch.int64)
            optimizer.zero_grad()

            output = model([traj, curr])

            loss = loss_fn(output, act)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Calculate training accuracy
            _, y_hat = torch.max(output, 1)
            correct_pred_train += (y_hat == act).sum().item()
            all_pred_train += act.size(0)

            running_loss_train += loss.item() * traj.size(0)

        train_acc = 100 * correct_pred_train / all_pred_train
        train_accuracy.append(train_acc)
        train_loss.append(running_loss_train / len(TrainLoader))

        model.eval()
        with torch.no_grad():
            for idx, data in enumerate(ValLoader, 0):
                traj, curr, act = data
                traj, curr, act = traj.to(device), curr.to(device), act.to(device)
                act = act.type(torch.int64)

                output = model([traj, curr])

                loss = loss_fn(output, act)

                # Calculate training accuracy
                _, y_hat = torch.max(output.data, 1)

                correct_pred_val += (y_hat == act).sum().item()
                all_pred_val += act.size(0)

                running_loss_val += loss.item() * traj.size(0)

            val_acc = 100 * correct_pred_val / all_pred_val
            val_accuracy.append(val_acc)
            val_loss.append(running_loss_val / len(ValLoader))

        print(
            f"| Epoch: {epoch} | Training Loss: {train_loss[-1]: .4f} % | Train Accuracy: {train_accuracy[-1]: .4f} % | Validation Accuracy: {val_accuracy[-1]: .4f} %"
        )
        print(
            "---------------------------------------------------------------------------------------"
        )

    print("Finished Training!")
    plt.plot(
        [epoch for epoch in range(EPOCHS)], train_accuracy, label="Training accuracy"
    )
    plt.plot(
        [epoch for epoch in range(EPOCHS)], val_accuracy, label="Validation accuracy"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.title("Training and Validation Accuracy")
    plt.savefig(
        f"ToMnetF/Results/Model/Experiment{EXPERIMENT_NO}/accuracy_experiment{EXPERIMENT_NO}"
    )
    plt.show()

    plt.plot([epoch for epoch in range(EPOCHS)], train_loss, label="Training loss")
    plt.plot([epoch for epoch in range(EPOCHS)], val_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.grid(True)
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(
        f"ToMnetF/Results/Model/Experiment{EXPERIMENT_NO}/loss_experiment{EXPERIMENT_NO}"
    )
    plt.show()


train()

print("Saving Model")
model.SaveModel(f"ToMnetF/Results/Model/Experiment{EXPERIMENT_NO}/ToMnetF.pt")
print("Model Saved - Bye!")
