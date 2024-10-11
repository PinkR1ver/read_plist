import pygame
import sys
import math
import argparse

def main(diagonal_inches, display_duration):
    # 初始化 Pygame
    pygame.init()
    pygame.mixer.init()

    # 获取屏幕信息并创建全屏窗口
    screen_info = pygame.display.Info()
    width, height = screen_info.current_w, screen_info.current_h
    screen = pygame.display.set_mode((width, height), pygame.FULLSCREEN)
    pygame.display.set_caption("视觉注意力训练")

    # 定义颜色
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)

    # 计算显示器的物理宽度（厘米）
    aspect_ratio = width / height
    diagonal_cm = diagonal_inches * 2.54
    width_cm = math.sqrt((diagonal_cm**2) / (1 + 1/aspect_ratio**2))

    # 计算每像素代表的厘米数
    cm_per_pixel = width_cm / width

    # 定义红点的位置（厘米）
    dot_positions_cm = [8.75, 17.5, 26.25, 43.75]

    # 转换厘米到像素
    def cm_to_pixels(cm):
        return int(cm / cm_per_pixel)

    # 计算红点位置（像素）
    dot_positions_px = [cm_to_pixels(pos) for pos in dot_positions_cm]

    # 定义小球半径（根据显示器尺寸调整）
    ball_diameter_cm = diagonal_cm * 0.020  # 红点直径为对角线长度的2%
    ball_radius = cm_to_pixels(ball_diameter_cm / 2)

    # 定义小球位置序列
    def calculate_positions(center_y):
        center_x = width // 2
        positions = [(center_x, center_y)]  # 中心点
        
        # 向右的点
        right_points = [(center_x + pos, center_y) for pos in dot_positions_px]
        positions.extend(right_points)
        
        positions.append((center_x, center_y))  # 回到中心
        
        # 向左的点
        left_points = [(center_x - pos, center_y) for pos in dot_positions_px]
        positions.extend(left_points)
        
        positions.append((center_x, center_y))  # 最后回到中心
        
        return positions

    # 初始化小球位置
    center_y = height // 2
    ball_positions = calculate_positions(center_y)

    # 加载声音
    beep_sound = pygame.mixer.Sound("beep.wav")  # 请确保你有一个名为beep.wav的音频文件

    # 主循环
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
                elif event.key == pygame.K_DOWN and adjusting:
                    center_y = min(height - ball_radius, center_y + 5)
                    ball_positions = calculate_positions(center_y)

        current_time = pygame.time.get_ticks()

        # 填充黑色背景
        screen.fill(BLACK)

        if adjusting:
            # 显示中心红点，可调整位置
            pygame.draw.circle(screen, RED, ball_positions[0], ball_radius)
        elif program_started:
            elapsed_time = current_time - start_time
            
            # 检查是否需要播放声音并变换位置
            if current_time >= next_beep_time:
                beep_sound.play()
                current_position = min(current_position + 1, len(ball_positions) - 1)
                next_beep_time += display_duration
                
                # 检查是否结束
                if current_position == len(ball_positions) - 1:
                    running = False
            
            # 绘制当前小球
            pygame.draw.circle(screen, RED, ball_positions[current_position], ball_radius)

        # 更新显示
        pygame.display.flip()

        # 控制帧率
        clock.tick(60)

    # 程序结束后等待几秒
    pygame.time.wait(2000)

    # 退出 Pygame
    pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="视觉注意力训练程序")
    parser.add_argument("--diagonal", type=float, required=True, help="显示器对角线尺寸（英寸）")
    parser.add_argument("--duration", type=int, default=5000, help="每个点的显示时间（毫秒）")
    args = parser.parse_args()

    main(args.diagonal, args.duration)