"""
Utilising RAY RLlib framework for training and testing the agent

Tensorboard can be used to visualise the training process
python main.py --logdir=$HOME/ray_results
"""

import ray
import ray.rllib.agents.ppo as ppo
import gym

class customExperimentClass:
    def __init__(self):
        ray.shutdown()
        ray.init(num_cpus=4, num_gpus=0)
        self.env = gym.make("Taxi-v3")
        
        self.config = ppo.DEFAULT_CONFIG.copy()
        self.config["num_workers"]=1
        self.config["env"] = "Taxi-v3"
        self.config["log_level"] = "WARN"

        self.agent = ppo.PPOTrainer(config=self.config)

    def tune(self, stopping_criteria):
        """
        Train an RLlib agent using tune until any of the configured stopping criteria is met.
            See https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run
        :return: Return the path to the saved agent (checkpoint) and tune's ExperimentAnalysis object
            See https://docs.ray.io/en/latest/tune/api_docs/analysis.html#experimentanalysis-tune-experimentanalysis
        """
        results = ray.tune.run(ppo.PPOTrainer,
                    verbose=1,
                    config=self.config,
                    stop=stopping_criteria,
                    checkpoint_freq=1,
                    keep_checkpoints_num=1,
                    checkpoint_score_attr='training_iteration',
                   )
        checkpoints = results.get_trial_checkpoints_paths(trial=results.get_best_trial('episode_reward_mean'),
                                                           metric='episode_reward_mean')
        # retriev the checkpoint path; we only have a single checkpoint, so take the first one
        checkpoint_path = checkpoints[0][0]
        return checkpoint_path, results
    
    def train(self, n_iterations):
        """
        """
        s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"

        for n in range(n_iterations):
          result = self.agent.train()
          file_name = self.agent.save()

          print(s.format(
            n + 1,
            result["episode_reward_min"],
            result["episode_reward_mean"],
            result["episode_reward_max"],
            result["episode_len_mean"],
            file_name
           ))
  
        return file_name, result
    
    def load(self, path):
        """
        Load a trained RLlib agent from the specified path. Call this before testing a trained agent.
        :param path: Path pointing to the agent's saved checkpoint (only used for RLlib agents)
        """
        self.agent.restore(path)

    def test(self):
        """Test trained agent for a single episode. Return the episode reward"""
        # instantiate env class
        env = self.env

        # run until episode ends
        episode_reward = 0
        done = False
        obs = env.reset()
        while not done:
            action = self.agent.compute_single_action(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward

        return episode_reward

    def model_details(self):
        """Return the model's summary"""
        # get the policy
        policy = self.agent.get_policy()
        # get the model
        model = policy.model
        # print model summary
        print(model.base_model.summary())

        return True

    def evaluate(self):
        """Return the model's summary"""
        self.config["evaluation_num_workers"]=1
        self.config["evaluation_config"]={
                                        "render_env": False,
                                        }

        self.agent = ppo.PPOTrainer(config=self.config)
        print(experiment.agent.evaluate())

        return True    
    
if __name__ == "__main__":
    experiment = customExperimentClass()
    checkpoint_path, results = experiment.tune({"training_iteration": 3, "episode_reward_mean": 0})
    # checkpoint_path, results = experiment.train(3)
    print(results)
    # checkpoint_path = "/Users/kylesaltmarsh/ray_results/PPO_2022-11-25_15-33-03/PPO_Taxi-v3_65677_00000_0_2022-11-25_15-33-03/checkpoint_000035"
    experiment.load(checkpoint_path)
    episode_reward = experiment.test()
    print("Episode reward:", episode_reward)
    experiment.evaluate()