import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pygame
import random
import time
import threading

st.set_page_config(
    page_title="Q-Learning Coin Hunter AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #0f1117;
    color: white;
}

.stButton>button {
    width: 100%;
    border-radius: 12px;
    height: 3em;
    background: linear-gradient(90deg,#2563eb,#7c3aed);
    color: white;
    border: none;
    font-weight: bold;
}

.metric-box {
    background: #161b22;
    padding: 15px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #2d3748;
}

.title {
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:white;
    margin-bottom:20px;
}

.grid-box {
    display:flex;
    justify-content:center;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Q-Learning Coin Hunter AI</div>', unsafe_allow_html=True)

if "trained" not in st.session_state:
    st.session_state.trained = False

if "q_table" not in st.session_state:
    st.session_state.q_table = None

if "best_reward" not in st.session_state:
    st.session_state.best_reward = -9999

if "training_data" not in st.session_state:
    st.session_state.training_data = None

with st.sidebar:
    st.header("⚙️ Settings")

    grid_size = st.slider("Grid Size", 6, 8, 6)
    episodes = st.slider("Episodes", 500, 2000, 1000, 100)
    learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1)
    gamma = st.slider("Gamma", 0.1, 0.99, 0.95)
    epsilon = st.slider("Epsilon", 0.01, 1.0, 1.0)
    epsilon_decay = st.slider("Epsilon Decay", 0.990, 0.9999, 0.995)

    train_btn = st.button("🚀 Train Model")
    run_btn = st.button("🎮 Run Trained Agent")
    reset_btn = st.button("♻️ Reset Training")

if reset_btn:
    st.session_state.trained = False
    st.session_state.q_table = None
    st.session_state.training_data = None
    st.session_state.best_reward = -9999
    st.rerun()

actions = [0, 1, 2, 3]

grid_placeholder = st.empty()

col1, col2, col3, col4 = st.columns(4)

episode_metric = col1.empty()
reward_metric = col2.empty()
coin_metric = col3.empty()
epsilon_metric = col4.empty()

progress_bar = st.progress(0)

status_text = st.empty()

chart_placeholder = st.empty()

def create_obstacles(size):
    obstacles = []

    for _ in range(size // 2):
        x = random.randint(0, size - 1)
        y = random.randint(0, size - 1)
        obstacles.append((x, y))

    return obstacles

def draw_grid(size, agent, coin, obstacles):
    grid = np.zeros((size, size))

    for ox, oy in obstacles:
        grid[ox][oy] = -1

    grid[coin[0]][coin[1]] = 2
    grid[agent[0]][agent[1]] = 1

    fig, ax = plt.subplots(figsize=(5,5))

    ax.imshow(grid)

    ax.set_xticks(np.arange(-.5, size, 1))
    ax.set_yticks(np.arange(-.5, size, 1))

    ax.grid(color='white', linestyle='-', linewidth=2)

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return fig

def get_state(agent, coin):
    return (
        agent[0],
        agent[1],
        coin[0],
        coin[1]
    )

def move_agent(agent, action, size):
    x, y = agent

    if action == 0:
        x -= 1
    elif action == 1:
        x += 1
    elif action == 2:
        y -= 1
    elif action == 3:
        y += 1

    return [x, y]

def run_pygame(q_table, size):
    pygame.init()

    cell_size = 80
    width = size * cell_size
    height = size * cell_size + 50

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Trained Agent")

    clock = pygame.time.Clock()

    font = pygame.font.SysFont("Arial", 24)

    agent = [0, 0]
    coin = [size - 1, size - 1]

    obstacles = create_obstacles(size)

    running = True
    episode = 1

    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        state = get_state(agent, coin)

        if state in q_table:
            action = np.argmax(q_table[state])
        else:
            action = random.choice(actions)

        new_agent = move_agent(agent, action, size)

        x, y = new_agent

        if x < 0 or x >= size or y < 0 or y >= size:
            new_agent = agent

        if tuple(new_agent) in obstacles:
            new_agent = agent

        agent = new_agent

        if agent == coin:
            episode += 1
            agent = [0, 0]

            while True:
                coin = [
                    random.randint(0, size - 1),
                    random.randint(0, size - 1)
                ]

                if tuple(coin) not in obstacles:
                    break

        screen.fill((15, 17, 23))

        for i in range(size):
            for j in range(size):

                rect = pygame.Rect(
                    j * cell_size,
                    i * cell_size,
                    cell_size,
                    cell_size
                )

                pygame.draw.rect(screen, (40,40,40), rect, 1)

                if (i, j) in obstacles:
                    pygame.draw.rect(screen, (120,120,120), rect)

                if [i, j] == coin:
                    pygame.draw.circle(
                        screen,
                        (255,215,0),
                        rect.center,
                        20
                    )

                if [i, j] == agent:
                    pygame.draw.rect(
                        screen,
                        (0,120,255),
                        (
                            rect.x + 15,
                            rect.y + 15,
                            50,
                            50
                        )
                    )

        fps = int(clock.get_fps())

        text = font.render(
            f"Episode: {episode} | FPS: {fps}",
            True,
            (255,255,255)
        )

        screen.blit(text, (10, height - 40))

        pygame.display.flip()

        clock.tick(10)

    pygame.quit()

if train_btn:

    size = grid_size

    q_table = {}

    rewards_history = []
    steps_history = []
    coins_history = []

    epsilon_value = epsilon

    total_coins = 0

    obstacles = create_obstacles(size)

    for ep in range(episodes):

        agent = [
            random.randint(0, size - 1),
            random.randint(0, size - 1)
        ]

        while tuple(agent) in obstacles:
            agent = [
                random.randint(0, size - 1),
                random.randint(0, size - 1)
            ]

        coin = [
            random.randint(0, size - 1),
            random.randint(0, size - 1)
        ]

        while tuple(coin) in obstacles:
            coin = [
                random.randint(0, size - 1),
                random.randint(0, size - 1)
            ]

        total_reward = 0
        steps = 0
        done = False

        max_steps = size * size * 2

        while not done:

            state = get_state(agent, coin)

            if state not in q_table:
                q_table[state] = np.zeros(4)

            if random.uniform(0,1) < epsilon_value:
                action = random.choice(actions)
            else:
                action = np.argmax(q_table[state])

            new_agent = move_agent(agent, action, size)

            reward = -0.1

            x, y = new_agent

            if x < 0 or x >= size or y < 0 or y >= size:
                reward = -1
                new_agent = agent

            if tuple(new_agent) in obstacles:
                reward = -1
                new_agent = agent

            if new_agent == coin:
                reward = 10 + max(0, 5 - steps * 0.1)
                total_coins += 1
                done = True

            next_state = get_state(new_agent, coin)

            if next_state not in q_table:
                q_table[next_state] = np.zeros(4)

            old_value = q_table[state][action]
            next_max = np.max(q_table[next_state])

            new_value = old_value + learning_rate * (
                reward + gamma * next_max - old_value
            )

            q_table[state][action] = new_value

            agent = new_agent

            total_reward += reward
            steps += 1

            fig = draw_grid(size, agent, coin, obstacles)

            grid_placeholder.pyplot(fig)

            plt.close(fig)

            episode_metric.metric("Episode", ep + 1)
            reward_metric.metric("Reward", round(total_reward,2))
            coin_metric.metric("Coins", total_coins)
            epsilon_metric.metric("Epsilon", round(epsilon_value,3))

            progress_bar.progress((ep + 1) / episodes)

            status_text.info(
                f"""
                Training AI Agent...
                
                Episode: {ep+1}/{episodes}
                Steps: {steps}
                Coins Collected: {total_coins}
                """
            )

            time.sleep(0.01)

            if steps >= max_steps:
                done = True

        rewards_history.append(total_reward)
        steps_history.append(steps)
        coins_history.append(total_coins)

        if total_reward > st.session_state.best_reward:
            st.session_state.best_reward = total_reward

        epsilon_value *= epsilon_decay

    moving_avg = []

    window = 20

    for i in range(len(rewards_history)):
        start = max(0, i - window)
        moving_avg.append(
            np.mean(rewards_history[start:i+1])
        )

    success_rate = (
        len([r for r in rewards_history if r > 0])
        / len(rewards_history)
    ) * 100

    st.session_state.trained = True
    st.session_state.q_table = q_table

    st.session_state.training_data = {
        "rewards": rewards_history,
        "steps": steps_history,
        "coins": coins_history,
        "episodes": list(range(1, len(rewards_history) + 1)),
        "moving_avg": moving_avg,
        "success_rate": success_rate
    }

    st.success("Training Completed!")

if st.session_state.training_data:

    data = st.session_state.training_data
    episode_numbers = data.get(
        "episodes",
        list(range(1, len(data["rewards"]) + 1))
    )

    st.markdown("## 📊 Training Analytics")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "🏆 Best Reward",
        round(st.session_state.best_reward,2)
    )

    col2.metric(
        "🎯 Success Rate",
        f"{data['success_rate']:.2f}%"
    )

    col3.metric(
        "🪙 Total Coins",
        data["coins"][-1]
    )

    fig0, ax0 = plt.subplots(figsize=(12,5))
    ax0.plot(
        episode_numbers,
        data["rewards"],
        color="#64748b",
        alpha=0.35,
        linewidth=1,
        label="Episode reward"
    )
    ax0.plot(
        episode_numbers,
        data["moving_avg"],
        color="#22c55e",
        linewidth=2.5,
        label="Moving average reward"
    )
    ax0.axhline(0, color="#94a3b8", linewidth=1, linestyle="--", alpha=0.6)
    ax0.set_title("Learning Progress by Episode")
    ax0.set_xlabel("Episode")
    ax0.set_ylabel("Reward")
    ax0.grid(True, alpha=0.25)
    ax0.legend()
    st.pyplot(fig0)
    plt.close(fig0)

    fig1, ax1 = plt.subplots(figsize=(10,4))
    ax1.plot(episode_numbers, data["rewards"])
    ax1.set_title("Total Reward per Episode")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    st.pyplot(fig1)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(10,4))
    ax2.plot(episode_numbers, data["steps"])
    ax2.set_title("Steps per Episode")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    st.pyplot(fig2)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(10,4))
    ax3.plot(episode_numbers, data["coins"])
    ax3.set_title("Coins Collected")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Coins")
    st.pyplot(fig3)
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(10,4))
    ax4.plot(episode_numbers, data["moving_avg"])
    ax4.set_title("Moving Average Reward")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Reward")
    st.pyplot(fig4)
    plt.close(fig4)

if run_btn:

    if st.session_state.trained:

        threading.Thread(
            target=run_pygame,
            args=(
                st.session_state.q_table,
                grid_size
            ),
            daemon=True
        ).start()

        st.success("Pygame visualization started!")

    else:
        st.error("Train the model first.")
