import torch
import torch.nn as nn
import torch.nn.functional as F

# A quick check for a GPU. Using a GPU is like casting Haste on your training process.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLPEmbedder(nn.Module):
    """
    A trusty Multi-Layer Perceptron (MLP). Think of it as a specialized translator.
    It takes a specific piece of information (like a character's combat stats)
    and converts it into a "thought vector" or "embedding" that the main brain
    can easily understand. It's a small but vital part of the thinking process.
    """
    def __init__(self, input_dim, output_dim=8, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # The first layer receives the raw data.
            nn.ReLU(),                       # An activation function to add some non-linear "creativity".
            nn.Linear(hidden_dim, output_dim),# The second layer produces the final thought vector.
            nn.ReLU()
        )
    def forward(self, x):
        # Just pass the data through the network. Easy peasy.
        return self.net(x)

class Policy(nn.Module):
    """
    This is the AI's brain! The Policy network is the master architect.
    It examines the entire party of heroes, considers which enemies have already been
    chosen, and then decides which monster to add to the encounter next. It's a complex
    process of slicing data, passing it to specialized translators (the MLPEmbedders),
    and then combining all those thoughts to make a final, informed decision.
    """
    def __init__(self, index_ranges, num_classes=26, numeric_out=8, cat1_out=2, cat2_out=4, cat3_out=4, synergy_out=8, hidden_dim=128, layer_norm=False):
        """
        The constructor builds the neural network's architecture.

        :param index_ranges: A dictionary mapping feature sets to their start/end indices in the input vector.
        :param num_classes: The number of unique monsters in our bestiary.
        :param ..._out: The output dimensions for our various MLP embedders.
        :param hidden_dim: The size of the hidden layer in the final decision-making MLP.
        :param layer_norm: A boolean to enable Layer Normalization, a technique to help stabilize learning.
        """
        super().__init__()
        self.index_ranges = index_ranges
        self.num_classes = num_classes
        self.stop_index = num_classes  # We add a special "STOP" action so the AI knows when it's done building the encounter.
        self.layer_norm = layer_norm

        # --- 1. Build the Specialist Translators (MLP Embedders) ---
        # We create a unique MLP for each slice of the character data.
        n_start, n_end = index_ranges["numerical"]
        self.numeric_mlp = MLPEmbedder(n_end - n_start, numeric_out, hidden_dim=32)
        
        c1_start, c1_end = index_ranges["cat_set_1"]
        self.cat1_mlp = MLPEmbedder(c1_end - c1_start, cat1_out, hidden_dim=16)

        c2_start, c2_end = index_ranges["cat_set_2"]
        self.cat2_mlp = MLPEmbedder(c2_end - c2_start, cat2_out, hidden_dim=16)

        c3_start, c3_end = index_ranges["cat_set_3"]
        self.cat3_mlp = MLPEmbedder(c3_end - c3_start, cat3_out, hidden_dim=16)

        # --- 2. The Synergy Translator ---
        # This MLP looks at the enemies already chosen and creates a "synergy" embedding.
        self.synergy_mlp = MLPEmbedder(num_classes, synergy_out, hidden_dim=32)

        # --- 3. The Final Decision Maker ---
        # First, calculate the size of the combined thought vector.
        row_emb_dim = numeric_out + cat1_out + cat2_out + cat3_out
        final_input_dim = (8 * row_emb_dim) + synergy_out # 8 heroes + synergy info

        # Now, build the final MLP that outputs the decision.
        self.fc1 = nn.Linear(final_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes + 1) # +1 for the STOP action
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, party_tensor, synergy_tensor):
        """
        The forward pass is where the thinking happens. Data flows through the network.

        :param party_tensor: A tensor representing the party of heroes.
        :param synergy_tensor: A tensor indicating which enemies have been picked so far.
        :return: A set of logits, representing the AI's confidence in picking each monster next.
        """
        batch_size = party_tensor.size(0)
        row_embeddings = []
        # Process each of the 8 potential hero slots in the party.
        for row_idx in range(8):
            row_data = party_tensor[:, row_idx, :]

            # Slice the data for the current hero and pass it through the appropriate translator.
            n_start, n_end = self.index_ranges["numerical"]
            numeric_emb = self.numeric_mlp(row_data[:, n_start:n_end].float())
            
            c1_start, c1_end = self.index_ranges["cat_set_1"]
            cat1_emb = self.cat1_mlp(row_data[:, c1_start:c1_end].float())

            c2_start, c2_end = self.index_ranges["cat_set_2"]
            cat2_emb = self.cat2_mlp(row_data[:, c2_start:c2_end].float())
            
            c3_start, c3_end = self.index_ranges["cat_set_3"]
            cat3_emb = self.cat3_mlp(row_data[:, c3_start:c3_end].float())

            # Combine all the translated thoughts for this one hero.
            row_emb = torch.cat([numeric_emb, cat1_emb, cat2_emb, cat3_emb], dim=1)
            row_embeddings.append(row_emb)

        # Stack all the hero thoughts together into one big party representation.
        party_emb = torch.stack(row_embeddings, dim=1).view(batch_size, -1)
        # Get the translated thought for the current encounter synergy.
        synergy_emb = self.synergy_mlp(synergy_tensor.float().detach())

        # Combine the party thoughts and synergy thoughts.
        x = torch.cat([party_emb, synergy_emb], dim=1)

        # Pass the final combined thought through the decision-making layers.
        x = F.relu(self.fc1(x))
        if self.layer_norm:
            x = self.ln(x)
        logits = self.fc2(x)
        return logits

def sample_enemies(policy_net, party_tensor, device, max_enemies=8):
    """
    This function lets the AI actually choose the monsters for an encounter.
    It's an iterative process: it picks one monster, adds it to the list,
    and then re-evaluates to pick the next one, until it decides to STOP.

    :param policy_net: The trained AI brain (Policy network).
    :param party_tensor: The tensor for the current party of heroes.
    :param device: The CPU or GPU we're running on.
    :param max_enemies: The maximum number of monsters allowed in the encounter.
    :return: A list of the chosen enemy indices and the log probabilities of those choices (for learning).
    """
    # This tensor keeps track of the monsters we've already picked.
    picks_count = torch.zeros((1, policy_net.num_classes), device=device)
    chosen_indices = []
    log_probs = [] # We need to store these to know how confident the AI was in its choices.

    for _ in range(max_enemies):
        # 1. Get the AI's current thoughts (logits).
        logits = policy_net(party_tensor, picks_count)
        # 2. Convert thoughts into probabilities.
        probs = F.softmax(logits, dim=-1)
        # 3. Create a probability distribution and sample an action (i.e., pick a monster).
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        action_idx = action.item()

        # 4. Record the log probability of the action. This is crucial for learning.
        log_probs.append(dist.log_prob(action))

        # 5. If the AI chose to STOP, the encounter is complete.
        if action_idx == policy_net.stop_index:
            break
        else:
            # 6. Otherwise, add the monster to our list and update the synergy tensor.
            chosen_indices.append(action_idx)
            with torch.no_grad(): # We don't want this update to affect the learning process.
                picks_count = picks_count.detach().clone()
                picks_count[0, action_idx] += 1

    return chosen_indices, log_probs

def reinforce_update(log_probs, reward, optimizer, device):
    """
    This is the learning step! The core of the REINFORCE algorithm.
    It looks at the choices the AI made (`log_probs`) and the outcome (`reward`).
    If the reward was good, it nudges the AI to make those choices more often.
    If the reward was bad, it nudges it to avoid those choices.

    :param log_probs: The list of log probabilities from `sample_enemies`.
    :param reward: The final reward from the combat simulation.
    :param optimizer: The PyTorch optimizer that updates the AI's weights.
    :param device: The CPU or GPU being used.
    """
    # The "loss" is the negative sum of log probabilities, scaled by the reward.
    # We want to maximize this value, which is why we use a negative sign (optimizers minimize).
    sum_log_probs = torch.stack(log_probs).sum()
    if not isinstance(reward, torch.Tensor):
        reward = torch.tensor(reward, dtype=torch.float32, device=device)
    loss = -sum_log_probs * reward.detach() # We detach the reward as it's a fixed value.

    # The standard PyTorch magic to update the network weights.
    optimizer.zero_grad() # Reset previous gradients.
    loss.backward()       # Calculate new gradients.
    optimizer.step()      # Apply the updates.