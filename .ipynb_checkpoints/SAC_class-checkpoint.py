#gittoken: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
from tkinter.tix import InputOnly
import torch as tor
from torch import nn
import numpy as np
from pathlib import Path
from collections import deque
import matplotlib.pyplot as plt
import cv2
import random, time, datetime, os, copy

import gym

#class defin=============================================================================

#Q関数蔵数
class DualQNetwork(nn.Module):
    """
    s,aを入力としてQ値を返す関数
    """

    def __init__(self, input_dim, unitdim = 256):
        super().__init__()

        self.Q_first = nn.Sequential(
            nn.Linear(input_dim, unitdim),
            nn.ReLU(),
            nn.Linear(unitdim, unitdim),
            nn.ReLU(),
            nn.Linear(unitdim,1)
        )

        self.Q_second = copy.deepcopy(self.Q_first)

        #targetは内包関数として実装。こっちの方がやりやすい
        self.target_Q_first = copy.deepcopy(self.Q_first)
        self.target_Q_second = copy.deepcopy(self.Q_first)

        #パラメータ固定
        for p in self.target_Q_first.parameters():
            p.requires_grid = False
        
        for p in self.target_Q_second.parameters():
            p.requires_grid = False

    def forward(self, states, action, mode):
        if mode == "online":
            return self.Q_first(tor.cat([states,action], dim = states.dim()-1)), self.Q_second(tor.cat([states,action], dim = states.dim()-1))
        elif mode == "target":
            return self.target_Q_first(tor.cat([states,action], dim = states.dim()-1)), self.target_Q_second(tor.cat([states,action], dim = states.dim()-1))
    
class GaussianPolicy(nn.Module):
    """
    ポリシーネットワーク
    ガウス分布を用いて平均と分散を出力すルネットワーク
    確率要素を逆伝播に入れないために、ガウス分布をノイズzを用いて表している
    また、値が無限に分布しているガウス分布を-1から1の間に入れるためにハイパボリックタンジェントをかけて圧縮している。
    """
    
    def __init__(self, input_dim, action_dim, use_cuda, cuda_num, unitdim = 256, logprob_bias = 1e-6):
        super().__init__()

        self.action_dim = action_dim
        self.logprob_bias = logprob_bias
        self.use_cuda = use_cuda
        self.cuda_num = cuda_num

        self.policy = nn.Sequential(#基本のネットワーク
            nn.Linear(input_dim, unitdim),
            nn.ReLU(),
            nn.Linear(unitdim, unitdim),
            nn.ReLU()
        )

        self.mean = nn.Sequential(#分散を導出するネットワークzを導出するため
            nn.Linear(unitdim, self.action_dim),
            nn.Tanh()
        )

        self.logstd = nn.Sequential(#logをかけた平均を出力するネットワーク
            nn.Linear(unitdim, self.action_dim)#活性化関数はなしか、平均だからいらんのね
        )

    def forward(self, states):#予測用の関数
        
        #ここでlogの分散をとるのは値を大きくするためなのか?

        return self.mean(self.policy(states)), self.logstd(self.policy(states))

    #pytorchは計算グラフで偏微分が取れるからtf.functionでデコレートする必要はないかな
    def sample_action(self, states):
        means, logstds = self(states)#ここちょっとCUDA関係が心配
        stds = tor.exp(logstds)#logを戻してる

        if self.use_cuda:
            z = tor.normal(0,1.0,size=means.shape,device=tor.device(self.cuda_num))
        else:
            z = tor.normal(0,1.0,size=means.shape)

        actions = means + stds * z

        #-1～1にスケールを縮小して、logをとった確率を導出している。
        #これ方策全体のエントロピーを導出しているからtorch.sumかけてるのか
        logprobs_gauss = self._compute_logprob_gauss(means, stds, actions)
        actions_squashed = tor.tanh(actions)
        logprobs_squashed = logprobs_gauss - tor.sum(
            tor.log(-tor.square(actions_squashed).add_(-1.0-self.logprob_bias)), axis=actions_squashed.dim()-1, keepdims=True
        )

        return actions_squashed, logprobs_squashed



    def _compute_logprob_gauss(self, means, stds, actions):#actionsのlog確率を導出する関数
        """
        ガウスの確率分布関数
        """
        logprob = -0.5 * np.log(2*np.pi)
        logprob += - tor.log(stds)
        logprob += - 0.5 * tor.square((actions - means) / stds)
        logprob = tor.sum(logprob, axis=logprob.dim()-1, keepdim=True)
        return logprob

class LogAlpha(nn.Module):
    """
    エントロピーの影響係数αのクラスSGDで最適化
    """
    def __init__(self, logalpha):
        super().__init__()
        self.logalpha = nn.Parameter(tor.tensor(logalpha), requires_grad = True)

class SAC:
    def __init__(self, state_dim, action_dim, save_dir, **kwargs):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

    #デフォルトパラメータ========
        def_cuda_num = 0                            #[-]使用GPU
        def_unitdim = 256                           #[-]ユニット数
        def_logprob_bias = 1e-6                     #[-]log確率を求める際のバイアス値
        def_maxlen = 100000                         #[-]キャッシュの最大長さ
        def_batch_size = 32                         #[-]バッチサイズ
        def_gamma = 0.9                             #[-]Q値減衰率 γ
        def_logalpha = 1.0                          #[-]エントロピー影響係数　α
        def_target_entropy = -0.5 * self.action_dim #[-]目標エントロピー　H
        def_q_lr = 1e-3                             #[-]Q値学習率
        def_policy_lr = 1e-3                        #[-]policy学習率
        def_alpha_lr = 3e-4                         #[-]alpha学習率
        def_rho = 0.995                             #[-]ターゲットQネットワークの更新係数

        def_save_every = 5e5                        #[-]ネットワークの保存頻度
        def_burnin = 1e4                            #[-]ネットワークの最小学習ステップ数
        def_learn_every = 3                         #[-]ネットワークの学習タイミング

    #ネットワーク定義==================
        self.use_cuda = tor.cuda.is_available()
        self.cuda_num = kwargs.get('cuda_num',def_cuda_num)

        self.q_net = DualQNetwork(self.state_dim+self.action_dim, kwargs.get('unitdim',def_unitdim))
        self.policy_net = GaussianPolicy(
            self.state_dim, self.action_dim, self.use_cuda, self.cuda_num,kwargs.get('unitdim',def_unitdim), kwargs.get('logprob_bias',def_logprob_bias)
            )
        self.log_alpha = LogAlpha(kwargs.get('logalpha',def_logalpha))
        
        if self.use_cuda:
            self.q_net = self.q_net.to(device="cuda")
            self.policy_net = self.policy_net.to(device="cuda")
            self.log_alpha = self.log_alpha.to(device="cuda")
    
    #行動系パラメータ================
        self.save_every = kwargs.get('save_every',def_save_every)
        self.burnin = kwargs.get('burnin',def_burnin)
        self.learn_every = kwargs.get('learn_every',def_learn_every)

        self.curr_step = 0.0#ステップ数定義

    #キャッシュパラメータ=============
        self.memory = deque(maxlen = kwargs.get('maxlen',def_maxlen))#tensor型スタックメモリCUDAの位置関係なしに格納可能
        self.batch_size = kwargs.get('batch_size',def_batch_size)#バッチサイズ定義

    #Q値パラメータ===================
        self.gamma = kwargs.get('gamma',def_gamma)
    
    #Q値更新パラメータ===============
        self.q_optimizer = tor.optim.Adam(self.q_net.parameters(), lr = kwargs.get('q_lr',def_q_lr))
        self.q_loss_fn = nn.SmoothL1Loss()#損失関数
        self.q_rho = kwargs.get('rho',def_rho)
    
    #policy更新パラメータ============
        self.policy_optimizer = tor.optim.Adam(self.policy_net.parameters(), lr = kwargs.get('policy_lr',def_policy_lr))

    #α更新パラメータ=================
        #αだけ確率勾配法。不安定ならAdamにしたろ
        self.alpha_optimizer = tor.optim.SGD(self.log_alpha.parameters(), lr = kwargs.get('alpha_lr',def_alpha_lr))
        self.target_entropy = kwargs.get('target_entropy',def_target_entropy)
    

    #関数定義=======================
    def act(self,state):
        """
        方策関数現在の状態から行動決定を行う
        """
        if self.use_cuda:
            state = tor.tensor(state,dtype=tor.float).cuda(self.cuda_num)
        else:
            state = tor.tensor(state,dtype=tor.float)

        action, _ = self.policy_net.sample_action(state)

        #これだとバッチがそのまま帰ってくるからsqueezeした方がいいかな？
        self.curr_step += 1
        return action

    def cache(self, state, next_state, action, reward, done):
        """
        実行した結果をself.memoryに保存
        """
        if self.use_cuda:
            state = tor.tensor(state,dtype = tor.float).cuda(self.cuda_num)
            next_state = tor.tensor(next_state,dtype = tor.float).cuda(self.cuda_num)
            action = tor.tensor(action,dtype = tor.float).cuda(self.cuda_num)
            reward = tor.tensor([reward]).cuda(self.cuda_num)
            done = tor.tensor([done]).cuda(self.cuda_num)

        else:
            state = tor.tensor(state,dtype = tor.float)
            next_state = tor.tensor(next_state,dtype = tor.float)
            action = tor.tensor(action,dtype = tor.float)
            reward = tor.tensor([reward])
            done = tor.tensor([done])

        self.memory.append((state, next_state, action, reward, done,))


    def recall(self):
        """
        メモリからバッチを呼び出し
        """
        batch = random.sample(self.memory, self.batch_size)#リスト型でバッチを作成
        state, next_state, action, reward, done = map(tor.stack, zip(*batch))#mapのためにリストにしているっぽい

        return state, next_state, action, reward, done#余計な次元を消している
    
    def update_policy(self, states):
        alpha = tor.exp(self.log_alpha.logalpha)
        selected_actions, logprobs = self.policy_net.sample_action(states)
        q_min = self.td_estimate(states, selected_actions)
        loss = -1 * tor.mean(q_min + -1 * alpha * logprobs)
        self.policy_optimizer.zero_grad()#勾配リセット
        loss.backward()
        self.policy_optimizer.step()

        #確率は数値として扱うため、計算グラフを切っている
        entropy_diff = -logprobs.detach() - self.target_entropy
        alpha_loss = tor.mean(tor.exp(self.log_alpha.logalpha) * entropy_diff)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return loss.item()

    def td_estimate(self, state, action):
        current_Q1, current_Q2 = self.q_net(state, action, "online")

        return tor.minimum(current_Q1, current_Q2)
    
    @tor.no_grad()
    def td_target(self, rewards, next_states, done):
        next_actions, next_logprob = self.policy_net.sample_action(next_states)
        target_Q1, target_Q2 = self.q_net(next_states, next_actions, "target")
        alpha = tor.exp(self.log_alpha.logalpha)

        target = rewards + (1-done.float())*self.gamma*(
            tor.minimum(target_Q1, target_Q2) + -1 * alpha * next_logprob
        )

        return target

    def update_Q(self, td_estimate, td_target):
        loss = self.q_loss_fn(td_estimate, td_target)
        self.q_optimizer.zero_grad()#勾配をゼロにリセット
        loss.backward()#勾配導出
        self.q_optimizer.step()#勾配より最適化

        #targetパラメータ更新
        for onl1_parm, onl2_parm, tar1_parm, tar2_parm in zip(
            self.q_net.Q_first.parameters(), self.q_net.Q_second.parameters(), 
            self.q_net.target_Q_first.parameters(), self.q_net.target_Q_second.parameters()
            ):
            tar1_parm = self.q_rho*tar1_parm + (1.0 - self.q_rho)*onl1_parm
            tar2_parm = self.q_rho*tar2_parm + (1.0 - self.q_rho)*onl2_parm

        return loss.item()#損失返却

    def save(self):
        save_path = (
            self.save_dir / f"ant_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        tor.save(
            dict(q_model = self.q_net.state_dict(), 
                 policy_model = self.policy_net.state_dict(), 
                 alpha_model = self.log_alpha.state_dict()
                 ),
            save_path,
        )
        print(f"AntNet saved to {save_path} at step {self.curr_step}")
    
    #管理関数
    def learn(self):
        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None, None
        
        if self.curr_step % self.learn_every != 0:
            return None, None, None
        
        state, next_state, action, reward, done = self.recall()

        td_est = self.td_estimate(state, action)

        td_tgt = self.td_target(reward, next_state, done)

        q_loss = self.update_Q(td_est, td_tgt)

        p_loss = self.update_policy(state)

        return (td_est.mean().item(), p_loss, q_loss)

class Logger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'alpha':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanPolicyLoss':>15}{'MeanQLoss':>15}"
                f"{'MeanQValue':>15}{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_p_losses_plot = save_dir / "p_loss_plot.jpg"
        self.ep_avg_q_losses_plot = save_dir / "q_loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        #動画保存系のパラメータ
        final_movie = save_dir / "finish_movie.mp4"
        width = 480
        height = 480

        fps = 30

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video = cv2.VideoWriter(str(final_movie), fourcc, float(fps), (width, height))


        # 指標の履歴
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_p_losses = []
        self.ep_avg_q_losses = []
        self.ep_avg_qs = []

        # reacord()が呼び出されるたびに追加される移動平均
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_p_losses = []
        self.moving_avg_ep_avg_q_losses = []
        self.moving_avg_ep_avg_qs = []

        # 現在のエピソードの指標
        self.init_episode()

        # 時間を記録
        self.record_time = time.time()

    def log_step(self, reward, p_loss, q_loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if q_loss:
            self.curr_ep_p_loss += p_loss
            self.curr_ep_q_loss += q_loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "エピソード終了時の記録"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_p_loss = 0
            ep_avg_q_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_p_loss = np.round(self.curr_ep_p_loss / self.curr_ep_loss_length, 5)
            ep_avg_q_loss = np.round(self.curr_ep_q_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_p_losses.append(ep_avg_p_loss)
        self.ep_avg_q_losses.append(ep_avg_q_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_p_loss = 0.0
        self.curr_ep_q_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, alpha, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_p_loss = np.round(np.mean(self.ep_avg_p_losses[-100:]), 3)
        mean_ep_q_loss = np.round(np.mean(self.ep_avg_q_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_p_losses.append(mean_ep_p_loss)
        self.moving_avg_ep_avg_q_losses.append(mean_ep_q_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Alpha {alpha} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean PolicyLoss {mean_ep_p_loss} - "
            f"Mean QLoss {mean_ep_q_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{alpha:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_p_loss:15.3f}{mean_ep_q_loss:15.3f}"
                f"{mean_ep_q:15.3f}{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_rewards", "ep_lengths", "ep_avg_p_losses","ep_avg_q_losses", "ep_avg_qs"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()

    def log_video(self, rgb_array):
        self.video.write(rgb_array)

    def release_video(self):
        self.video.release()

#testcode
"""    
save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)
state = np.random.rand(10)
next_state = np.random.rand(10)
action = np.random.rand(5)
reward = random.random()
done = False  

test_ant = SAC(len(state), len(action), save_dir, learn_every=1, save_every=1, burnin = 0)
test_logger = Logger(save_dir)

td_tgt = test_ant.td_target(tor.rand(32,1), tor.rand(32,10), tor.zeros(32,1))

act = test_ant.act(state)

for i in range(0,32):
    state = np.random.rand(10)
    next_state = np.random.rand(10)
    action = np.random.rand(5)
    reward = random.random()
    
    test_ant.cache(state, next_state, action, reward, done)

q, p_loss, q_loss =  test_ant.learn()

test_logger.log_step(reward, p_loss, q_loss, q)

test_logger.log_episode()

test_logger.record(episode = int(1000), 
                   alpha = float(tor.exp(test_ant.log_alpha.logalpha)), 
                   step=int(test_ant.curr_step)
                   )
"""



