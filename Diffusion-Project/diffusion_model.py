
from ex1.Diffusion.helpers_and_metrics import *

# Setting reproducibility
SEED = 379
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
TRAINING_SAMPLES_COUNT = 3000  # increaet this form 3000

SAMPLES_DIM = 2
DENOISER_TRAINING_LEARNING_RATE = 0.007
DENOISER_TRAINING_EPOCHS = 1000
DENOISER_TRAINING_BATCH_SIZE = 128


# time_steps->T
# scheduler->σ

#######################################################################################################################
class Denoiser(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # #flattern
        self.linear_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            # nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            # nn.Dropout(0.1),
            nn.Linear(hidden_dim, 8),
            nn.LeakyReLU(),
            # nn.Dropout(0.1),
            nn.Linear(8, output_dim))

    def forward(self, x, time_step_embedding, class_embedding=None):
        time_step = time_step_embedding.float()  # convert long -> float before concatenating

        if class_embedding is not None:
            class_embedding = class_embedding.float()
            x = torch.cat([x, time_step, class_embedding], dim=-1)
        else:
            x = torch.cat([x, time_step], dim=-1)  # Condition on timestep by concatenating
        x = self.linear_layer(x)
        return x


class DiffusionDenoiser(nn.Module):
    def __init__(self, time_steps: int, time_embedding: int, samples_dim: int, denoiser_path: str = None):
        super(DiffusionDenoiser, self).__init__()
        self.time_step = time_steps
        self.time_embedding = time_embedding
        self.samples_dim = samples_dim
        self.denoiser_path = denoiser_path

        # loading denoiser
        denoiser_input_dim = time_embedding + samples_dim
        if denoiser_path is None:
            self.denoiser = Denoiser(input_dim=denoiser_input_dim, hidden_dim=16,
                                     output_dim=samples_dim)
        else:
            self.denoiser = self.load_model()

        # time steps embedding
        self.embedding = nn.Embedding(num_embeddings=time_steps,
                                      embedding_dim=time_embedding)  # one time embedding for all sample

    def forward(self, input_data, time_step):
        time_step_embedding = self.embedding(time_step.long())
        return self.denoiser(input_data, time_step_embedding)  # Pass through denoiser

    def load_model(self):
        self.denoiser = torch.load(self.denoiser_path)

    def save_model(self):
        torch.save(self.denoiser, self.denoiser_path)

    def train_denoiser(self, dataloader, optimizer, scheduler, loss_function=nn.MSELoss(),
                       epochs=DENOISER_TRAINING_EPOCHS):
        '''
        The Reverse Process - Training Denoiser
        return: losses for visualization
        '''
        self.denoiser.train()  # switch the model to training mode
        losses = []
        for epoch in range(epochs):
            for datapoints in dataloader:
                # forward process:
                # 1. Sample Gaussian noise ε ∼ N (0, I)
                noise = torch.randn(*datapoints.shape)
                # 2. Sample a random time t ∈ [0, T] T=timestep
                t = torch.rand(size=(datapoints.shape[0], 1))
                # 3. Obtain xt using xt=x0+σ2(t)·ε ε∼N(0,I)
                xt = datapoints + scheduler(t) * noise  # noisy points xt
                # 4. Backpropagate objective
                optimizer.zero_grad()
                prediction = self.forward(xt, t)
                loss = loss_function(prediction, noise)

                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        return losses

    def evaluate_denoiser(self, dataloader, scheduler, loss_function=nn.MSELoss()):
        self.denoiser.eval()  # switch the model to evaluation mode
        validation_losses = []
        with torch.no_grad():  # disable gradients computation
            for datapoints in dataloader:
                # 1. Sample Gaussian noise ε ∼ N (0, I)
                noise = torch.randn(*datapoints.shape)

                # 2. Sample a random time t ∈ [0, T] T=timestep
                t = torch.rand(size=(datapoints.shape[0], 1))

                # 3. Obtain xt using Eq. 4
                xt = datapoints + scheduler(t) * noise

                # Compute objective: l2(D(xt,t),ε)
                prediction = self.denoiser(xt, t)
                loss = loss_function(prediction, noise)

                validation_losses.append(loss.item())

        return validation_losses  # return the average loss

    # 2.1.3 The Reverse Process - DDIM Sampling

    def DDIM_sampling(self, scheduler, samples_num: int = TRAINING_SAMPLES_COUNT,
                      samples_dim: int = SAMPLES_DIM, fixed_point=None, trajectory=None):
        '''
        The Reverse Process - DDIM Sampling
        :param denoiser: denoiser neural network
        :param samples_num: number of samples
        :param samples_dim: dimension of each sample (2) for 2D-datapoints
        :return: predicted sample vector z
        '''
        if fixed_point == None:
            # Sample z ∼ N (0, I)
            z = torch.randn((samples_num, samples_dim))
        else:
            z = fixed_point

        self.denoiser.eval()

        # Iterate from t = 1 to 0 with step dt
        dt = 1 / self.time_step
        for t in np.arange(1, 0, -dt):

            t = torch.tensor(t, dtype=torch.float32).repeat(samples_num, 1)

            # Estimate noise:
            estimated_noise = self.forward(z, t)
            if t[0].item() == 1.0:
                print(estimated_noise[:3])

            # Estimate denoise x_hat
            x_hat = z - scheduler(t) * estimated_noise

            # Calculate the score function
            score_z = (x_hat - z) / (scheduler(t) ** 2)
            # Update z with the reverse process
            dz = derived_scheduler(t) * scheduler(t) * score_z * dt  # closed formula for reverse process
            z = z + dz
            if trajectory != None:
                trajectory.append(z.detach().numpy())
        # print(z[:2])
        return trajectory if trajectory else z

    def estimate_probability(self, x, scheduler, num_samples=1000):
        '''
        The Reverse Process - Probability Estimation
        :param x: point to estimate
        :param scheduler: noise scheduler function
        :param num_samples: qty of samples used for estimation
        :return: ELBO log(p(x))
        '''
        # Number of time steps
        T = self.time_step

        # 1. Randomize many possible noise and time combinations
        noise = torch.randn(num_samples, *x.shape)
        t = torch.rand(num_samples, *x.shape)  # x,shape = [2,1] #TODO: check if this is right

        # 2. Perform a forward process for all the combinations starting from x as the input
        xt = x + scheduler(t) * noise

        # 3. Estimate x0 for all combinations
        x0_hat = self.denoiser(xt, t)

        # 4. Compute SNR differences for the sampled t values
        dt = torch.ones_like(t) / T  # Assume constant dt for simplicity
        SNR_diff = scheduler(t - dt.abs()) - scheduler(t)

        # Compute the squared L2 distance ||x - x0_hat||²
        l2_distance = torch.sum((x - x0_hat) ** 2, dim=-1)

        # Compute the expectation over the noise and time samples
        expectation = torch.mean(SNR_diff * l2_distance)

        # 5. Average the results and multiply by T
        LT = (T * expectation) / 2

        # 6. -LT(x) is the lower bound for log p(x)
        log_p_x = -LT

        return log_p_x


class Cond_DiffusionDenoiser(nn.Module):
    def __init__(self, time_steps: int, time_embedding_size: int, samples_dim: int, num_classes: int,
                 class_embedding_size: int, denoiser_path: str = None):
        super(Cond_DiffusionDenoiser, self).__init__()
        self.time_step = time_steps
        self.time_embedding_size = time_embedding_size
        self.class_embedding_size = class_embedding_size
        self.samples_dim = samples_dim
        self.num_classes = num_classes
        self.denoiser_path = denoiser_path

        # class embedding
        self.class_embedding = nn.Embedding(num_embeddings=self.num_classes, embedding_dim=self.class_embedding_size)
        # time steps embedding
        self.embedding = nn.Embedding(num_embeddings=time_steps,
                                      embedding_dim=time_embedding_size)

        # loading denoiser
        denoiser_input_dim = self.time_embedding_size + self.samples_dim + self.class_embedding_size
        if denoiser_path is None:
            self.denoiser = Denoiser(input_dim=denoiser_input_dim, hidden_dim=16,
                                     output_dim=samples_dim)
        else:
            self.denoiser = torch.load(denoiser_path)

    def forward(self, input_data, time_step, class_idx):
        time_step_embedding = self.embedding(time_step.long()).squeeze(1)
        class_embedding = self.class_embedding(class_idx.long())
        return self.denoiser(input_data, time_step_embedding, class_embedding)

    def load_model(self):
        self.denoiser = torch.load(self.denoiser_path)

    def save_model(self, path):
        torch.save(self.denoiser, path)

    def train_denoiser(self, dataloader, optimizer, scheduler, loss_function=nn.MSELoss(),
                       epochs=DENOISER_TRAINING_EPOCHS, batch_size=DENOISER_TRAINING_BATCH_SIZE):
        '''
        The Reverse Process - Training Denoiser
        return: losses for visualization
        '''
        self.denoiser.train()  # switch the model to training mode
        losses = []
        for epoch in range(epochs):
            for samples, classes in dataloader:
                # forward process:
                # 1. Sample Gaussian noise ε ∼ N (0, I)
                noise = torch.randn(*samples.shape)
                # 2. Sample a random time t ∈ [0, T] T=timestep
                t = torch.rand(size=(samples.shape[0], 1))
                # 3. Obtain xt using xt=x0+σ2(t)·ε ε∼N(0,I)
                xt = samples + scheduler(t) * noise  # noisy points xt
                # 4. Backpropagate objective
                optimizer.zero_grad()
                prediction = self.forward(xt, t, classes)
                loss = loss_function(prediction, noise)

                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        return losses

    def evaluate_denoiser(self, dataloader, scheduler, loss_function=nn.MSELoss()):
        self.denoiser.eval()  # switch the model to evaluation mode
        validation_losses = []
        with torch.no_grad():  # disable gradients computation
            for datapoints in dataloader:
                # 1. Sample Gaussian noise ε ∼ N (0, I)
                noise = torch.randn(*datapoints.shape)

                # 2. Sample a random time t ∈ [0, T] T=timestep
                t = torch.rand(size=(datapoints.shape[0], 1))

                # 3. Obtain xt using Eq. 4
                xt = datapoints + scheduler(t) * noise

                # Compute objective: l2(D(xt,t),ε)
                prediction = self.denoiser(xt, t)
                loss = loss_function(prediction, noise)

                validation_losses.append(loss.item())

        return validation_losses  # return the average loss

    # 2.1.3 The Reverse Process - DDIM Sampling

    def DDIM_sampling(self, scheduler, samples_num: int,
                      samples_dim: int, sample_points: int,sample_classes: torch.Tensor, trajectory=None):
        '''
        The Reverse Process - DDIM Sampling
        :param denoiser: denoiser neural network
        :param samples_num: number of samples
        :param samples_dim: dimension of each sample (2) for 2D-datapoints
        :return: predicted sample vector z
        '''

        z = sample_points
        # Iterate from t = 1 to 0 with step dt
        dt = 1 / self.time_step
        self.denoiser.eval()
        for t in np.arange(1, 0, -dt):
            # we are interested in how the distribution changes over time
            # thus we will pass the same t to all our samples
            # if t==
            t = torch.tensor(t, dtype=torch.float32).repeat(samples_num, 1)

            # Estimate noise:

            estimated_noise = self.forward(z, t, sample_classes)

            if t[0].item() == 1.0:
                print(estimated_noise[:3])

            # Estimate denoise x_hat
            x_hat = z - scheduler(t) * estimated_noise

            # Calculate the score function
            score_z = (x_hat - z) / (scheduler(t) ** 2)
            # Update z with the reverse process
            dz = derived_scheduler(t) * scheduler(t) * score_z * dt  # closed formula for reverse process
            z = z + dz
            if trajectory != None:
                trajectory.append(z.detach().numpy())
        # print(z[:2])
        return trajectory if trajectory else z

    def estimate_probability(self, x, scheduler, num_samples=1000):
        '''
        The Reverse Process - Probability Estimation
        :param x: point to estimate
        :param scheduler: noise scheduler function
        :param num_samples: qty of samples used for estimation
        :return: ELBO log(p(x))
        '''
        # Number of time steps
        T = self.time_step

        # 1. Randomize many possible noise and time combinations
        noise = torch.randn(num_samples, *x.shape)
        t = torch.rand(num_samples, *x.shape)  # x,shape = [2,1] #TODO: check if this is right

        # 2. Perform a forward process for all the combinations starting from x as the input
        xt = x + scheduler(t) * noise

        # 3. Estimate x0 for all combinations
        x0_hat = self.denoiser(xt, t)

        # 4. Compute SNR differences for the sampled t values
        dt = torch.ones_like(t) / T  # Assume constant dt for simplicity
        SNR_diff = scheduler(t - dt.abs()) - scheduler(t)

        # Compute the squared L2 distance ||x - x0_hat||²
        l2_distance = torch.sum((x - x0_hat) ** 2, dim=-1)

        # Compute the expectation over the noise and time samples
        expectation = torch.mean(SNR_diff * l2_distance)

        # 5. Average the results and multiply by T
        LT = (T * expectation) / 2

        # 6. -LT(x) is the lower bound for log p(x)
        log_p_x = -LT

        return log_p_x


class Conditional_dataset(Dataset):
    def __init__(self, data_samples):
        self.samples = data_samples
        self.class_dict = {
            0: [[-1, -0.6], [0, 1]],  # top-left rectangle
            1: [[-0.6, -0.2], [0, 1]],  # top rectangle second from left
            2: [[-0.2, 0.2], [0, 1]],  # top rectangle in the middle
            3: [[0.2, 0.6], [0, 1]],  # top rectangle second from right
            4: [[0.6, 1], [0, 1]],  # top-right rectangle
            5: [[-1, -0.6], [-1, 0]],  # bottom-left rectangle
            6: [[-0.6, -0.2], [-1, 0]],  # bottom rectangle second from left
            7: [[-0.2, 0.2], [-1, 0]],  # bottom rectangle in the middle
            8: [[0.2, 0.6], [-1, 0]],  # bottom rectangle second from right
            9: [[0.6, 1], [-1, 0]],  # bottom-right rectangle
        }
        self.labels = self._create_class_labels(data_samples)
        self.data_distribution = self._get_data_distribution()


    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

    def __len__(self):
        return len(self.samples)

    def _create_class_labels(self, data_samples):
        class_labels = []
        for sample in data_samples:
            for key, value in self.class_dict.items():
                if value[0][0] <= sample[0] <= value[0][1] and value[1][0] <= sample[1] <= value[1][1]:
                    class_labels.append(key)
                    break
        return torch.tensor(class_labels)

    def get_class_points(self, class_idx):
        return self.samples[self.labels == class_idx]

    def _get_data_distribution(self):
        return self.labels.bincount().float() / len(self.labels)
