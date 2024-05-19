import cv2
import numpy as np
import math
import sys
sys.setrecursionlimit(640*480+1) # 최대 재귀 한도 설정

file_name_prefix = "kgs" # 파일 이름의 첫 세글자로 사용할 이니셜
curr_file_index = 1

window_name = "20192113 KIM GISOO"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

input_image = cv2.imread("./input.jpg", cv2.IMREAD_COLOR)
drawing_board = input_image.copy() # 추후 관심영역 이미지 저장을 위해 원본 이미지와 도형을 그릴 이미지를 분리하여 관리




# 모드 관련 변수, 메소드
curr_mode = 0 # 현재 모드 상태 저장하는 변수
mode_dict = {
    0: "NULL",
    1: "Ellipse",
    2: "Polygon"
} # 모드 값을 문자열로 변경하기 위한 딕셔너리
def draw_mode_text(image): # 좌측 상단에 현재 모드를 표시하는 함수
    global curr_mode
    cv2.putText(image, mode_dict[curr_mode], (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA)

def render_window(image): # 이미지의 좌측 상단에 현재 상태 텍스트를 추가하여 렌더링하는 함수
    render_image = image.copy()
    draw_mode_text(render_image)
    cv2.imshow(window_name, render_image)
render_window(drawing_board) # 초기 상태 렌더링 위해 호출

def mode_changer(image, next_mode): # 입력된 키에 따라 모드를 변경하는 함수
    global curr_mode, drawing_board
    if curr_mode == next_mode:
        curr_mode = 0
    else:
        curr_mode = next_mode
    render_window(drawing_board)




def save_image(mask): # 이미지 파일로 저장하는 함수
    global input_image, curr_file_index

    orig_b, orig_g, orig_r = cv2.split(input_image)
    roi_b, roi_g, roi_r = cv2.bitwise_and(orig_b, mask), cv2.bitwise_and(orig_g, mask), cv2.bitwise_and(orig_r, mask)
    roi = cv2.merge((roi_b, roi_g, roi_r)) # 원본 이미지와 MASK AND연산을 통하여 이미지에서 ROI만 추출

    cv2.namedWindow("roi", cv2.WINDOW_AUTOSIZE) # 추출된 roi 표시
    cv2.imshow("roi", roi)

    new_file_name = "{}{:04}.jpg".format(file_name_prefix, curr_file_index)
    cv2.imwrite(new_file_name, roi)
    curr_file_index += 1




# Ellipse 관련 메소드
def draw_ellipse(image, mask, start_pos, end_pos, color): # 타원 그리기 함수
    first_x, first_y = end_pos
    second_x, second_y = start_pos

    dist_x = max(first_x, second_x) - min(first_x, second_x)
    dist_y = max(first_y, second_y) - min(first_y, second_y)
    axes = (dist_x // 2, dist_y // 2) # 그릴 타원의 axes를 구하기 위한 연산

    center_x, center_y = (-1, -1)
    if first_x > second_x:
        center_x = first_x - axes[0]
    else:
        center_x = second_x - axes[0]
    if first_y > second_y:
        center_y = first_y - axes[1]
    else:
        center_y = second_y - axes[1]
    center = (center_x, center_y) # 그릴 타원의 center를 구하기 위한 연산

    cv2.ellipse(image, center, axes, 0, 0, 360, color, 2, cv2.LINE_AA) # 이미지에 빨간 테두리 타원 그리기
    if mask is not None:
        cv2.ellipse(mask, center, axes, 0, 0, 360, (255, 255, 255), -1) # mask에 내부가 칠해진 타원 그리기

last_pointed_pos = (-1, -1)
ellipse_mask = None
def ellipse_drawer(mouse_type, pos): # 타원 모드 처리 함수
    global last_pointed_pos, ellipse_mask, drawing_board

    if mouse_type == 0: # 좌클릭 했을 시
        last_pointed_pos = pos # 클릭 좌표 저장
        ellipse_mask = np.zeros(drawing_board.shape[:2], dtype="uint8") # 타원 mask 생성

    elif mouse_type == 1: # 좌클릭 땠을 시
        draw_ellipse(drawing_board, ellipse_mask, last_pointed_pos, pos, (0, 0, 255)) # 클릭 시작점부터 클릭 땐 위치까지의 확정된 원 그리기 + 마스크 완성
        render_window(drawing_board)

        save_image(ellipse_mask)

        last_pointed_pos = (-1, -1)
        ellipse_mask = None

    elif mouse_type == 2: # 드래그 중일 시
        dragging_image = drawing_board.copy() # 확정되지 않은 원을 그리기 위해 현재 이미지 복사
        draw_ellipse(dragging_image, None, last_pointed_pos, pos, (0, 0, 255)) # 클릭 시작점부터 현재 커서 위치까지 원 그리기
        render_window(dragging_image)




# Polygon 관련 메소드
def get_dist(p1, p2): #  점과 점 사이 거리 계산
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))
def is_invalid_polygon(poses): # 시계 반대방향인지 필터링하는 함수
    a, b, c = poses
    vector_ab = np.array((b[0] - a[0], b[1] - a[1]), dtype="float32")
    vector_bc = np.array((c[0] - b[0], c[1] - b[1]), dtype="float32")

    # AB와 BC의 외적의 결과가 양수이면 시계 방향에 있음
    if np.cross(vector_ab, vector_bc) > 0:
        return True
    return False

def fill_mask(mask, pos, visited): # DFS 탐색을 이용하여 테두리 바깥쪽 전부 하얀색으로 변경
    cy, cx = pos
    for oy, ox in ((-1, 0), (0, -1), (1, 0), (0, 1)):
        ny, nx = cy + oy, cx + ox

        if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
            if mask[ny][nx] > 0:
                continue
            if (ny, nx) not in visited:
                visited.add((ny, nx))
                mask[ny][nx] = 255
                fill_mask(mask, (ny, nx), visited)
def set_mask(mask): # 테두리만 그려져 있는 mask의 내부를 채우는 함수
    fill_mask(mask, (0, 0), set())
    mask = cv2.bitwise_not(mask) # 테두리 바깥쪽을 하얀색으로 변경한 마스크를 반전
    return mask

points = []
polygon_mask = None
def polygon_drawer(pos):
    global points, polygon_mask, drawing_board

    is_polygon_completed = False
    first_pos = (-1, -1)
    last_pos = (-1, -1)
    if len(points) > 0:
        first_pos = points[0]
        last_pos = points[-1]

    points.append(pos)
    if len(points) == 1: # 첫 점 일때 초기화
        polygon_mask = np.zeros(drawing_board.shape[:2], dtype="uint8")
        return
    if len(points) >= 3:
        if (is_invalid_polygon(points[-3:])): # 마지막 세 점이 다각형의 조건을 만족하지 않으면 무시
            points.pop()
            return

        if get_dist(first_pos, pos) <= 20: # 첫 점과의 거리가 20이하일 때 첫 점과 잇고 마지막 점 무시
            pos = first_pos
            is_polygon_completed = True

    cv2.line(drawing_board, last_pos, pos, (0, 0, 255), 2, cv2.LINE_AA) # 화면에 선 긋기
    cv2.line(polygon_mask, last_pos, pos, (255, 255, 255), 1) # mask에 선 긋기
    render_window(drawing_board)

    if is_polygon_completed:
        polygon_mask = set_mask(polygon_mask)

        save_image(polygon_mask)

        points = []
        polygon_mask = None




# Mouse Callback
def onMouse(event, x, y, flags, param): # onMouse Callback 함수
    global curr_mode

    if event == cv2.EVENT_LBUTTONDOWN:
        if curr_mode == 1:
            ellipse_drawer(0, (x, y))
        elif curr_mode == 2:
            polygon_drawer((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        if curr_mode == 1:
            ellipse_drawer(1, (x, y))
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if curr_mode == 1:
            ellipse_drawer(2, (x, y))
cv2.setMouseCallback(window_name, onMouse, drawing_board) # Mouse Callback 등록




# 키를 입력받기 위한 Loop
while True:
    key = cv2.waitKeyEx(0)
    if key == -1:
        break
    elif key == 49: # 입력된 키가 1일 때
        mode_changer(drawing_board, 1)
    elif key == 50: # 입력된 키가 2일 때
        mode_changer(drawing_board, 2)
cv2.destroyAllWindows()

