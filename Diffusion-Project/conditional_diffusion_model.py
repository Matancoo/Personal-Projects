from ex1.Diffusion.diffusion_model import *
from ex1.Diffusion.helpers_and_metrics import *


## Conditional Diffusion Denoiser
# Training
training_samples = sample_datapoints(a=-1, b=1, samples_num=TRAINING_SAMPLES_COUNT, samples_dim=SAMPLES_DIM)
cond_data = Conditional_dataset(training_samples)
print(cond_data.data_distribution)

dataloader = DataLoader(cond_data, batch_size=DENOISER_TRAINING_BATCH_SIZE, shuffle=True)
cond_denoiser = Cond_DiffusionDenoiser(time_steps=1000,
                                       time_embedding_size=16,
                                       samples_dim=SAMPLES_DIM,
                                       num_classes= len(cond_data.class_dict),
                                       class_embedding_size=16,
                                       denoiser_path=None)
optimizer = torch.optim.Adam(cond_denoiser.parameters(), lr=DENOISER_TRAINING_LEARNING_RATE)
train_losses = cond_denoiser.train_denoiser(dataloader,optimizer,exp_scheduler)
print(np.mean(train_losses))
cond_denoiser.save_model(path='denoiser_conditional.pt')

df1 = pd.DataFrame(data={'training_step': range(1, len(train_losses) + 1), 'loss': train_losses})
fig1 = px.line(df1,
               x='training_step',
               y='loss',
               title='loss function over the training batches of the denoiser')
fig1.show()

# Q1
colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'black', 'cyan', 'magenta']
fig = go.Figure()
for class_idx in range(10):
    class_points = cond_data.get_class_points(class_idx)
    fig.add_trace(
        go.Scatter(
            x=class_points[:, 0],
            y=class_points[:, 1],
            mode='markers',
            name=f'Class {class_idx}',
            marker=dict(size=5, color=colors[class_idx]
                        )
        ))
fig.show()

# Q2
# get sample for each class
cond_denoiser = Cond_DiffusionDenoiser(time_steps=1000,
                              time_embedding_size=16,
                              samples_dim=SAMPLES_DIM,
                              num_classes=len(cond_data.class_dict),
                              class_embedding_size=16,
                              denoiser_path='denoiser_conditional.pt')
fig = go.Figure()
for class_idx in cond_data.class_dict.keys():
    sample = cond_data.get_class_points(class_idx)[0]
    sample = sample[None,:]
    trajectory = cond_denoiser.DDIM_sampling(exp_scheduler,
                                        samples_num=1,
                                        samples_dim=2,
                                        sample_points=sample,
                                        sample_classes = torch.tensor([class_idx]),
                                        trajectory=[sample])
    fig.add_trace(go.Scatter(x=[point[0][0] for point in trajectory], y=[point[0][1] for point in trajectory],
                             mode='lines', name=f'Trajectory {class_idx + 1}'))
fig.update_layout(title='Denoising trajectories of points in classes',
                  xaxis_title='X',
                  yaxis_title='Y',
                  legend_title='Trajectory')
fig.show()

# Q4
fig = go.Figure()
training_samples = sample_datapoints(a=-1, b=1, samples_num=1000, samples_dim=SAMPLES_DIM)
cond_data = Conditional_dataset(training_samples)
print(cond_data.data_distribution)
cond_denoiser = Cond_DiffusionDenoiser(time_steps=1000,
                                       time_embedding_size=16,
                                       samples_dim=SAMPLES_DIM,
                                       num_classes=len(cond_data.class_dict),
                                       class_embedding_size=16,
                                       denoiser_path='model/denoiser_conditional.pt')
z = cond_denoiser.DDIM_sampling(exp_scheduler,
                                samples_num=1000,
                                samples_dim=2,
                                sample_points=cond_data.samples,
                                sample_classes=cond_data.labels,
                                trajectory=None)
# Create scatter plot and add to figure
colors = np.array(['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'black', 'cyan', 'magenta'])
colors_array = colors[cond_data.labels.numpy()]
fig.add_trace(
    go.Scatter(
        x=z[:, 0].detach().numpy(),
        y=z[:, 1].detach().numpy(),
        mode="markers",
        marker=dict(
            size=6,
            color=colors_array,  # set color equal to a variable
        ),
    ),
)

fig.show()


# Q6


