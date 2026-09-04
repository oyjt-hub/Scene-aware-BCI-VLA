import os
import sys
import time
import pygame
import numpy as np
import scipy.io as scio
import requests
import base64
import io
import csv
import json
import cv2  # 🌟 新增 cv2 用于处理轮廓

from dataServer import DataServerThread
from triggerBox import TriggerBox
from creat_raw_data import CreateRawData
from fbcca import FBCCA

# ================= 配置区域 =================
LINUX_SERVER_IP = "172.19.5.252"
LINUX_SERVER_PORT = 8000
TRIGGER_COM_PORT = "COM3"
SAVE_PATH = r'D:\BCI_Data'


# ============================================

class BCIClient:
    def __init__(self, subject_name):
        self.decoder = FBCCA()
        self.save_data = []
        self.save_label = []
        self.bci_log = []
        self.trial_id = 0
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)
        self.save_mat = os.path.join(SAVE_PATH, f'subject_{subject_name}_{int(time.time())}.mat')
        self.save_csv = os.path.join(SAVE_PATH, f'bci_log_{int(time.time())}.csv')

        self.thread_data_server = None
        self.fps = 60
        self.stimulate_time = 4.0

        pygame.init()
        infoObject = pygame.display.Info()
        self.screen_w, self.screen_h = infoObject.current_w, infoObject.current_h
        self.surface = pygame.display.set_mode((self.screen_w, self.screen_h), pygame.FULLSCREEN)
        pygame.display.set_caption("BCI Dynamic Client")
        self.font = pygame.font.Font(None, 50)

    def connect_eeg(self):
        print("🧠 正在连接 Neuracle...")
        neuracle = dict(device_name='Neuracle', hostname='127.0.0.1', port=8712, srate=1000, n_chan=12)
        self.thread_data_server = DataServerThread(
            device=neuracle['device_name'], n_chan=neuracle['n_chan'],
            srate=neuracle['srate'], t_buffer=6
        )
        notconnect = self.thread_data_server.connect(hostname=neuracle['hostname'], port=neuracle['port'])
        if notconnect:
            raise ConnectionError("❌ 无法连接脑电仪，请确认博睿康软件已开启网络广播！")
        self.thread_data_server.start()
        self.thread_data_server.Daemon = True
        print("✅ 脑电连接成功！")

    def request_ui_from_server(self):
        self.surface.fill((0, 0, 0))
        text = self.font.render("Waiting for Linux Server to send scene data...", True, (255, 255, 255))
        self.surface.blit(text, text.get_rect(center=(self.screen_w // 2, self.screen_h // 2)))
        pygame.display.update()

        url = f"http://{LINUX_SERVER_IP}:{LINUX_SERVER_PORT}/get_bci_scene"
        try:
            print("📡 正在向 Linux 请求场景数据...")
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data["status"] == "success" and len(data["labels"]) > 0:
                    print(f"✅ 成功获取场景，包含 {len(data['labels'])} 个目标")
                    return data
        except Exception as e:
            print(f"❌ 请求 Linux 服务器失败: {e}")
        return None

    def send_result_to_server(self, selected_label):
        url = f"http://{LINUX_SERVER_IP}:{LINUX_SERVER_PORT}/submit_bci_result"
        try:
            requests.post(url, json={"selected_target": selected_label}, timeout=5)
            print("✅ 结果发送成功！")
        except Exception as e:
            print(f"❌ 发送结果失败: {e}")

    def save_bci_log(self):
        if not self.bci_log: return
        with open(self.save_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['trial', 'timestamp', 'selection_time_s', 'decode_latency_ms', 'num_targets', 'pred_index',
                        'selected_freq', 'selected_label', 'eeg_samples'])
            w.writerows(self.bci_log)

    def run(self):
        self.connect_eeg()
        #     try:
        #         triggerbox = TriggerBox(TRIGGER_COM_PORT)
        #         print(f"✅ TriggerBox 连接成功 ({TRIGGER_COM_PORT})")
        #     except Exception as e:
        #         print(f"⚠️ TriggerBox 连接失败: {e}")
        #         triggerbox = None
        triggerbox = None
        clock = pygame.time.Clock()

        while True:
            # 1. 待机界面
            self.surface.fill((0, 0, 0))
            ready_text = self.font.render("Waiting for signal to begin", True, (255, 255, 0))
            self.surface.blit(ready_text, ready_text.get_rect(center=(self.screen_w // 2, self.screen_h // 2)))
            pygame.display.update()

            waiting_for_enter = True
            while waiting_for_enter:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        scio.savemat(self.save_mat, {'data': self.save_data, 'label': self.save_label})
                        self.save_bci_log()
                        self.thread_data_server.stop()
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN and (
                            event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER):
                        waiting_for_enter = False
                clock.tick(30)

            # 2. 请求数据
            scene_data = self.request_ui_from_server()
            if not scene_data:
                time.sleep(1)
                continue

            labels = scene_data["labels"]
            bboxes = scene_data["bboxes"]
            masks_b64 = scene_data.get("masks", [])


            img_data = base64.b64decode(scene_data["image_b64"])
            bg_image_raw = pygame.image.load(io.BytesIO(img_data))
            img_w, img_h = bg_image_raw.get_size()

            scale = min(self.screen_w / img_w, self.screen_h / img_h)
            new_w, new_h = int(img_w * scale), int(img_h * scale)
            x_offset, y_offset = (self.screen_w - new_w) // 2, (self.screen_h - new_h) // 2

            bg_image_scaled = pygame.transform.smoothscale(bg_image_raw, (new_w, new_h))
            canvas = pygame.Surface((self.screen_w, self.screen_h))
            canvas.fill((0, 0, 0))
            canvas.blit(bg_image_scaled, (x_offset, y_offset))


            dark_overlay = pygame.Surface((self.screen_w, self.screen_h), pygame.SRCALPHA)
            dark_overlay.fill((0, 0, 0, 160))  # 160/255 的黑色遮罩
            canvas.blit(dark_overlay, (0, 0))

            flash_surfaces = []

            for i, b64 in enumerate(masks_b64):

                nparr = np.frombuffer(base64.b64decode(b64), np.uint8)
                mask = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)


                resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                full_mask = np.zeros((self.screen_h, self.screen_w), dtype=np.uint8)
                full_mask[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_mask


                current_area = np.sum(full_mask > 127)

                MIN_AREA_THRESHOLD = self.screen_w * self.screen_h * 0.015

                if 0 < current_area < MIN_AREA_THRESHOLD:

                    kernel = np.ones((15, 15), np.uint8)

                    iterations = min(int(np.sqrt(MIN_AREA_THRESHOLD / current_area)) + 1, 8)
                    full_mask = cv2.dilate(full_mask, kernel, iterations=iterations)


                cv_canvas = np.zeros((self.screen_h, self.screen_w, 4), dtype=np.uint8)


                cv_canvas[full_mask > 127] = [255, 255, 255, 255]


                cv_canvas = cv_canvas.transpose([1, 0, 2])
                flash_surf = pygame.surfarray.make_surface(cv_canvas[:, :, :3])
                flash_surf.set_colorkey((0, 0, 0))
                flash_surf.set_alpha(255)
                flash_surfaces.append(flash_surf)


            num_objects = len(labels)
            freqs = [2.0 + i * (8.0 / max(1, num_objects - 1)) for i in range(num_objects)]
            self.decoder.get_reference_signals(freqs, round(self.stimulate_time, 2))


            self.surface.blit(canvas, (0, 0))
            pygame.display.update()
            time.sleep(1.0)

            print("🧠 开始闪烁刺激...")
            stim_begin = time.perf_counter()
            self.trial_id += 1
            self.thread_data_server.ResetBuffer()
            if triggerbox:
                triggerbox.output_event_data(1)

            start_time = time.time()


            while time.time() - start_time < self.stimulate_time:
                elapsed = time.time() - start_time
                self.surface.blit(canvas, (0, 0))  # 重绘暗色背景

                for i, flash_surf in enumerate(flash_surfaces):
                    freq = freqs[i]
                    if int(elapsed * freq * 2) % 2 == 0:

                        self.surface.blit(flash_surf, (0, 0))


                        bbox = bboxes[i]
                        x1 = int(bbox[0] * scale) + x_offset
                        y1 = int(bbox[1] * scale) + y_offset
                        bw = int((bbox[2] - bbox[0]) * scale)

                        text_surf = self.font.render(f"{labels[i]} ({freq:.1f}Hz)", True, (0, 255, 0))
                        text_rect = text_surf.get_rect(center=(x1 + bw // 2, max(30, y1 - 25)))
                        text_bg = pygame.Surface((text_rect.width + 10, text_rect.height + 10), pygame.SRCALPHA)
                        text_bg.fill((0, 0, 0, 200))
                        self.surface.blit(text_bg, (text_rect.x - 5, text_rect.y - 5))
                        self.surface.blit(text_surf, text_rect)

                pygame.display.update()
                clock.tick(self.fps)

            self.surface.fill((0, 0, 0))
            text = self.font.render("Decoding EEG signals...", True, (255, 255, 255))
            self.surface.blit(text, text.get_rect(center=(self.screen_w // 2, self.screen_h // 2)))
            pygame.display.update()

            time.sleep(0.5)
            data = self.thread_data_server.GetBufferData()
            data_new = data[0:9, :]
            data_raw = CreateRawData(data_new, 1000)
            data_np = data_raw.get_data()

            decode_samples_1000hz = int(self.stimulate_time * 1000) - 200
            data_1000hz = data_np[:, 125: 125 + decode_samples_1000hz]
            data_250hz = data_1000hz[:, ::4]
            actual_samples_250hz = data_250hz.shape[1]
            actual_decode_sec = actual_samples_250hz / 250.0
            self.decoder.get_reference_signals(freqs, actual_decode_sec)
            decode_start = time.perf_counter()
            pred_idx = self.decoder.fbcca(data_250hz, 7)
            decode_latency_ms = (time.perf_counter() - decode_start) * 1000

            if 0 <= pred_idx < len(labels):
                selected_label = labels[pred_idx]
            else:
                selected_label = labels[0]

            selection_time = time.perf_counter() - stim_begin
            self.bci_log.append([self.trial_id, time.time(), selection_time, decode_latency_ms, len(labels), pred_idx,
                                 freqs[pred_idx] if 0 <= pred_idx < len(freqs) else -1, selected_label,
                                 actual_samples_250hz])
            print(f"🎯 解码完成！选中目标: {selected_label}")
            self.save_data.append(data)
            self.save_label.append(selected_label)

            self.surface.fill((0, 0, 0))


            status_surf = self.font.render("Object Selected!", True, (0, 255, 0))  # 绿色高亮
            status_rect = status_surf.get_rect(center=(self.screen_w // 2, self.screen_h // 2 - 40))


            result_surf = self.font.render(f"Result: {selected_label}", True, (255, 255, 255))  # 白色文字
            result_rect = result_surf.get_rect(center=(self.screen_w // 2, self.screen_h // 2 + 40))

            self.surface.blit(status_surf, status_rect)
            self.surface.blit(result_surf, result_rect)
            pygame.display.update()
            # ==========================================================

            self.send_result_to_server(selected_label)
            time.sleep(3)


if __name__ == '__main__':
    client = BCIClient(subject_name='test_01')
    try:
        client.run()
    except KeyboardInterrupt:
        print("\n⚠️ 收到强制中断信号 (Ctrl+C)，程序准备退出...")
    except Exception as e:
        print(f"\n❌ 程序发生异常退出: {e}")
    finally:
        # 无论程序是怎么退出的，这里都会执行！
        print("💾 正在紧急保存数据...")

        # 保存 mat 数据
        if client.save_data and client.save_label:
            scio.savemat(client.save_mat, {'data': client.save_data, 'label': client.save_label})
            print(f"✅ 脑电数据已保存至: {client.save_mat}")

        # 保存 csv 日志
        if client.bci_log:
            client.save_bci_log()
            print(f"✅ 日志数据已保存至: {client.save_csv}")

        # 安全关闭子线程和 Pygame
        if client.thread_data_server:
            client.thread_data_server.stop()
        pygame.quit()
        print("👋 程序已安全关闭。")