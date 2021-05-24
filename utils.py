from skimage.transform import resize
import imageio

def learn(session, replay_memory, main_dqn, target_dqn, batch_size, gamma):
    """
    Extrage un minilot din memoria de replay, apoi calculeaza valoarea Q pentru reteaua tinta 
    apoi calculeaza parametrii in reteaua principala
        session: sesiunea tensorflow
        replay_memory: memoria de replay
        main_dqn: reteaua principala
        target_dqn: reteaua tinta
        batch_size: dimensiunea unui minilot
        gamma: factorul de discount
    Returns:
        loss: pierderea pentru un minilot
    """
    # ia un minilot din memoria de replay
    states, actions, rewards, new_states, done_flags = replay_memory.get_minibatch()    
    # determina, pentru fiecare experienta din minilot, care actiune este mai buna in state-ul urmator, in reteaua principala
    arg_q_max = session.run(main_dqn.best_action, feed_dict={main_dqn.input:new_states})
    # estimeaza valorile Q in reteaua tinta pentru fiecare experienta din minilot
    q_vals = session.run(target_dqn.q_values, feed_dict={target_dqn.input:new_states})
    double_q = q_vals[range(batch_size), arg_q_max]
    # determina valoarea tinta pentru q cu ecuatia Bellman
    target_q = rewards + (gamma*double_q * (1-done_flags))
    # Executa o coborare in gradient negativ
    loss, _ = session.run([main_dqn.loss, main_dqn.update], feed_dict={main_dqn.input:states, main_dqn.target_q:target_q, main_dqn.action:actions})
    return loss



def generate_gif(frame_number, frames_for_gif, reward, path):
    """
        Genereaza un fisier gif din frame-urile primite
            frame_number: numarul pasului curent
            frames_for_gif: o lista de cadre RGB  (210, 160, 3)
            reward: recompensa totala a episodului
            path: calea unde salvam animatia
    """
    for idx, frame_idx in enumerate(frames_for_gif): 
        frames_for_gif[idx] = resize(frame_idx, (420, 320, 3), preserve_range=True, order=0).astype(np.uint8)
        
    imageio.mimsave(f'{path}{"ATARI_frame_{0}_reward_{1}.gif".format(frame_number, reward)}', frames_for_gif, duration=1/30)



def clip_reward(reward):
    if reward > 0:
        return 1
    elif reward == 0:
        return 0
    else:
        return -1