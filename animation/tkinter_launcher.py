import tkinter as tk
from tkinter import ttk, messagebox
import subprocess

def start_experiment():
    diagonal = diagonal_entry.get()
    duration = duration_entry.get()
    dot_size = dot_size_entry.get()
    command = f"python standardization_animation.py --diagonal {diagonal} --duration {duration} --dot_size {dot_size}"
    subprocess.Popen(command, shell=True)
    root.quit()

def show_instructions():
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

def show_credits():
    credits_text = """
    视觉注意力训练程序

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

# 创建主窗口
root = tk.Tk()
root.title("视觉注意力训练配置")

# 创建菜单栏
menubar = tk.Menu(root)
root.config(menu=menubar)

# 创建 Help 菜单
help_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="帮助", menu=help_menu)
help_menu.add_command(label="使用说明", command=show_instructions)
help_menu.add_command(label="关于", command=show_credits)

# 创建并放置组件
ttk.Label(root, text="显示器对角线尺寸（英寸）:").grid(column=0, row=0, padx=5, pady=5)
diagonal_entry = ttk.Entry(root)
diagonal_entry.grid(column=1, row=0, padx=5, pady=5)
diagonal_entry.insert(0, "24")

ttk.Label(root, text="每个点的显示时间（毫秒）:").grid(column=0, row=1, padx=5, pady=5)
duration_entry = ttk.Entry(root)
duration_entry.grid(column=1, row=1, padx=5, pady=5)
duration_entry.insert(0, "5000")

ttk.Label(root, text="红点大小（显示器对角线的百分比）:").grid(column=0, row=2, padx=5, pady=5)
dot_size_entry = ttk.Entry(root)
dot_size_entry.grid(column=1, row=2, padx=5, pady=5)
dot_size_entry.insert(0, "2")

start_button = ttk.Button(root, text="开始实验", command=start_experiment)
start_button.grid(column=0, row=3, columnspan=2, pady=10)

# 运行主循环
root.mainloop()