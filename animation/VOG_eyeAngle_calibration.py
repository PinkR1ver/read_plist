import tkinter as tk
from tkinter import ttk, messagebox
import pygame
import sys
import math
import os

def resource_path(relative_path):
    """ 获取资源的绝对路径 """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, relative_path)

class VisualAttentionTrainer:
    def __init__(self, root):
        self.root = root
        self.root.title("视觉注意力训练配置")
        self.create_widgets()
        self.create_menu()
        self.center_y = None  # 添加这行来存储调整后的位置

    def create_widgets(self):
        ttk.Label(self.root, text="显示器对角线尺寸（英寸）:").grid(column=0, row=0, padx=5, pady=5)
        self.diagonal_entry = ttk.Entry(self.root)
        self.diagonal_entry.grid(column=1, row=0, padx=5, pady=5)
        self.diagonal_entry.insert(0, "24")

        ttk.Label(self.root, text="每个点的显示时间（毫秒）:").grid(column=0, row=1, padx=5, pady=5)
        self.duration_entry = ttk.Entry(self.root)
        self.duration_entry.grid(column=1, row=1, padx=5, pady=5)
        self.duration_entry.insert(0, "5000")

        ttk.Label(self.root, text="红点大小（显示器对角线的百分比）:").grid(column=0, row=2, padx=5, pady=5)
        self.dot_size_entry = ttk.Entry(self.root)
        self.dot_size_entry.grid(column=1, row=2, padx=5, pady=5)
        self.dot_size_entry.insert(0, "1")

        start_button = ttk.Button(self.root, text="开始实验", command=self.start_experiment)
        start_button.grid(column=0, row=3, columnspan=2, pady=10)

    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="使用说明", command=self.show_instructions)
        help_menu.add_command(label="关于", command=self.show_credits)

    def show_instructions(self):
        instructions = """
        使用说明:
        1. 输入显示器对角线尺寸(英寸)、每个点的显示时间(毫秒)和红点大小(显示器对角线的百分比)。
        2. 点击"开始实验"按钮启动实验。
        3. 在实验窗口中:
           - 使用上下箭头键调整红点的垂直位置
           - 按空格键开始实验
           - 按 ESC 键退出实验
        4. 实验过程中,请集中注意力跟随红点移动。
        """
        messagebox.showinfo("使用说明", instructions)

    def show_credits(self):
        credits_text = """
        VOG Eye Angle Calibration

        开发者: JudeW
        版本: 1.0

        使用的库:
        - Tkinter
        - Pygame

        特别感谢:
        Doc. Shi

        © 2024 版权所有
        """
        messagebox.showinfo("关于", credits_text)

    def start_experiment(self):
        diagonal = float(self.diagonal_entry.get())
        duration = int(self.duration_entry.get())
        dot_size = float(self.dot_size_entry.get())
        self.root.withdraw()  # 隐藏主窗口
        self.run_experiment(diagonal, duration, dot_size)
        self.root.deiconify()  # 实验结束后重新显示主窗口

    def run_experiment(self, diagonal_inches, display_duration, dot_size_percent):
        pygame.init()
        pygame.mixer.init()

        screen_info = pygame.display.Info()
        width, height = screen_info.current_w, screen_info.current_h
        screen = pygame.display.set_mode((width, height), pygame.FULLSCREEN)
        pygame.display.set_caption("视觉注意力训练")
        pygame.mouse.set_visible(False)

        BLACK = (0, 0, 0)
        RED = (255, 0, 0)

        aspect_ratio = width / height
        diagonal_cm = diagonal_inches * 2.54
        width_cm = math.sqrt((diagonal_cm**2) / (1 + 1/aspect_ratio**2))
        cm_per_pixel = width_cm / width

        dot_positions_cm = [8.75, 17.5, 26.25, 43.75]
        dot_positions_px = [int(pos / cm_per_pixel) for pos in dot_positions_cm]

        ball_diameter_cm = diagonal_cm * (dot_size_percent / 100)
        ball_radius = int(ball_diameter_cm / (2 * cm_per_pixel))

        def calculate_positions(center_y):
            center_x = width // 2
            positions = [(center_x, center_y)]
            positions.extend([(center_x + pos, center_y) for pos in dot_positions_px])
            positions.append((center_x, center_y))
            positions.extend([(center_x - pos, center_y) for pos in dot_positions_px])
            positions.append((center_x, center_y))
            return positions

        if self.center_y is None:
            self.center_y = height // 2  # 如果是第一次运行,设置为屏幕中间
        center_y = self.center_y
        ball_positions = calculate_positions(center_y)

        beep_sound_path = resource_path("beep.wav")
        if os.path.exists(beep_sound_path):
            beep_sound = pygame.mixer.Sound(beep_sound_path)
        else:
            print(f"警告：找不到音频文件 {beep_sound_path}")
            beep_sound = None

        running = True
        adjusting = True
        program_started = False
        current_position = 0
        start_time = 0
        next_beep_time = 0

        clock = pygame.time.Clock()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE and adjusting:
                        adjusting = False
                        program_started = True
                        start_time = pygame.time.get_ticks()
                        next_beep_time = start_time + display_duration
                    elif event.key == pygame.K_UP and adjusting:
                        center_y = max(ball_radius, center_y - 5)
                        ball_positions = calculate_positions(center_y)
                        self.center_y = center_y  # 更新存储的位置
                    elif event.key == pygame.K_DOWN and adjusting:
                        center_y = min(height - ball_radius, center_y + 5)
                        ball_positions = calculate_positions(center_y)
                        self.center_y = center_y  # 更新存储的位置

            current_time = pygame.time.get_ticks()

            screen.fill(BLACK)

            if adjusting:
                pygame.draw.circle(screen, RED, ball_positions[0], ball_radius)
            elif program_started:
                if current_time >= next_beep_time:
                    if beep_sound:
                        beep_sound.play()
                    current_position = min(current_position + 1, len(ball_positions) - 1)
                    next_beep_time += display_duration
                    if current_position == len(ball_positions) - 1:
                        running = False
                
                pygame.draw.circle(screen, RED, ball_positions[current_position], ball_radius)

            pygame.display.flip()
            clock.tick(60)

        pygame.time.wait(2000)
        pygame.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = VisualAttentionTrainer(root)
    root.mainloop()
