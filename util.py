import matplotlib.pyplot as plt

def seq_vs_loss(loss_lis,batch,epoch,l=None):
    if l == None:
        l = len(loss_lis)
    
    seq_len = list(range(1, l + 1))

    plt.figure(figsize=(10, 5))  # Set the figure size (optional)
    plt.plot(seq_len, loss_lis, marker='o', color='b', linestyle='-', label='Loss per sequence length')
    plt.title(f'seq_len vs loss, batch : {batch}, epoch : {epoch}')  # Title of the plot
    plt.xlabel('Epoch')  # X-axis label
    plt.ylabel('Loss')  # Y-axis label
    plt.grid(True)  # Show grid lines
    plt.legend()  # Show legend
    plt.savefig(f'model_loads/plots/seq_len_vs_loss_batch_{batch}_epoch_{epoch}.png',dpi=300)
    