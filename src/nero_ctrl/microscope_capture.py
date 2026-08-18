# -*- coding: utf-8 -*-
"""
microscope_capture.py — USB 현미경 자동 촬영 + 직사각형 마커 포즈 측정 (Windows·Ubuntu / OpenCV)

Tunable Surface Parallel Gripper Pick&Place 정확도 실험용 바닥 비전.
시편 바닥의 검은 8x4mm 직사각형 마커(흰/초록 등 밝은 바탕, 플러시)를 검출해
(dx, dy, theta)를 계산한다. 검출은 네 변의 서브픽셀 에지 직선 피팅 + 교점 방식.
OS(Windows/Linux)는 자동 감지되어 캡처 백엔드·노출 규약이 알아서 분기된다.

필요 패키지:
    pip install opencv-python numpy
    (모니터 없는 서버/SSH 환경이면 대화형 프리뷰 불가 → --interval 모드 사용)

사용법:
    python microscope_capture.py --probe            # 카메라 인덱스 탐색(최초 1회)
    python microscope_capture.py                    # 대화형: 프리뷰(실시간 측정) + 키보드 트리거
    python microscope_capture.py --index 1          # 내장캠 있는 노트북이면 보통 1
    python microscope_capture.py --interval 20 --count 100   # 무인 자동 촬영(드리프트 테스트)
    python microscope_capture.py --calc-scale captures/.../calib_001_120000.png --cb 9x6 --pitch 2.0
                                                    # 체커보드 캡처에서 μm/px 스케일 계산 → calib.json 기록
    python microscope_capture.py --show-calib       # 저장된 영점·스케일 확인
    python microscope_capture.py --clear-zero       # 영점만 삭제(스케일 유지)
    python microscope_capture.py --ignore-zero      # 이번 실행만 저장된 영점 무시

캘리브 파일(calib.json):
    z 키로 잡은 영점과 --calc-scale 로 구한 스케일이 이 파일에 저장되어
    프로그램을 껐다 켜도 유지된다. 구조:
        {"scale": {"um_per_px": ..., "source": "checkerboard", ...},
         "zero":  {"dx_px": ..., "dy_px": ..., "theta_deg": ..., "time": ...}}
    영점은 "카메라와 스테이지가 그대로일 때"만 유효하다. 현미경을 옮기거나
    초점(z 높이)을 바꿨으면 z 를 다시 눌러 갱신할 것.

대화형 키:
    SPACE : 촬영/저장/측정 (현재 모드로 시행번호 자동 증가)
    r / s : 모드 전환 rigid / soft
    z     : 영점(zero) 촬영 — 이 순간의 마커 포즈가 Δ의 기준이 되며 calib.json 에 저장
            (이후 프리뷰·저장본에 'Z:' 줄로 영점 대비 Δdx,Δdy,Δr,Δθ 표시, CSV에도 기록)
    c     : 체커보드 캘리브레이션 촬영
    d     : 직전 촬영 삭제(오트리거 취소, 번호 되돌림 / 영점이면 직전 영점 복원)
    q     : 종료

좌표/각도 규약:
    * dx, dy 는 "이미지 중심" 기준(px). z(영점) 촬영 시 그 마커 포즈가 기준으로
      저장되어, 이후 화면 'Z:' 줄과 CSV(d_dx_mm 등)에 영점 대비 Δ값이 함께 기록된다.
      → 변위 검증·본실험 분석은 반드시 이 Δ값(영점 차감 후)을 사용할 것.
        화면 괄호 안 (x.xxx mm) 는 이미지 중심 기준 생값이라 시작 위치 오프셋이 섞인다.
    * theta 는 마커 긴 축이 이미지 x축과 이루는 각(도), (-90, +90] 범위.
      이미지 y축이 아래 방향이므로 화면상 시계방향 회전이 +.
    * mm 환산: SCALE_UM_PER_PX(체커보드 값) 우선, 없으면 긴 변 8mm 기준 자동 스케일.

주의:
    * 다른 프로그램이 카메라 점유 중이면 열리지 않음
      (Windows: HiView 등 / Ubuntu: cheese, guvcview 등). 반드시 종료 후 실행.

우분투 참고:
    * 카메라는 /dev/video* 로 잡히며 USB 카메라 하나가 노드 2개(영상+메타데이터)를
      만든다. 내장캠 있는 노트북이면 현미경은 보통 index 2 — --probe 로 확인.
    * 권한 오류(열 수 없음) 시: sudo usermod -aG video $USER  후 재로그인.
    * 수동 노출(EXPOSURE) 값의 단위가 OS마다 다름 — 아래 설정 주석 참조.

로봇 컨트롤러 연동(완전 자동 트리거)이 필요해지면:
    이 파일을 import 하여 open_camera()를 한 번 부른 뒤, place 완료 신호가
    올 때마다(pyserial/socket 등) capture_and_save()를 호출하면 된다.
"""

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

IS_WINDOWS  = platform.system() == "Windows"
CAP_BACKEND = cv2.CAP_DSHOW if IS_WINDOWS else cv2.CAP_V4L2   # Ubuntu: V4L2
CAP_NAME    = "DSHOW" if IS_WINDOWS else "V4L2"
FOURCC_STR  = "YUY2" if IS_WINDOWS else "YUYV"                # 같은 포맷의 OS별 명칭

# ----------------------------- 촬영 설정 -----------------------------
WIDTH, HEIGHT = 640, 480   # 이 센서의 실화소
N_BURST  = 5               # 트리거당 촬영 후 평균할 프레임 수
N_FLUSH  = 5               # 촬영 직전 버퍼에 고인 옛 프레임 버리는 수
EXPOSURE = None            # None=자동노출. 수동 값의 단위는 OS별로 다름:
                           #  Windows(DSHOW): log2초 음수 정수, 예 -6 ≈ 1/64 s
                           #  Ubuntu(V4L2)  : 100μs 단위 정수, 예 156 ≈ 15.6 ms (기기별 상이)
SAVE_RAW = True            # 평균본 외에 원본(단일 프레임) 1장도 같이 저장
SAVE_ANNOTATED = True      # 검출 결과가 그려진 이미지도 저장(*_det.png)
OUT_ROOT = Path("captures")
CALIB_FILE = Path("calib.json")   # 영점(zero)·스케일 저장 파일. --calib 로 변경 가능.
                                  #  z 키로 잡은 영점이 여기 기록되어 재실행해도 유지된다.
WINDOW   = "microscope (SPACE:capture r/s:mode z:zero c:calib d:undo q:quit)"

# ------------------- 마커 검출 설정 (검은 8x4 직사각형) -------------------
MARKER_LONG_MM  = 8.0      # 마커 긴 변 실치수 → mm 자동 스케일에 사용
                           #  ※ 자동 스케일 사용 시 이 값의 오차가 그대로 %오차가 된다.
                           #    캘리퍼스 실측값을 넣을 것 (예: 7.94, 8.12 ...)
MARKER_SHORT_MM = 4.0
SCALE_UM_PER_PX = None     # 체커보드 캘리브레이션 값(μm/px). --calc-scale 로 계산해 입력.
                           #  입력 시 자동 스케일보다 우선. 캘리브 때와 z(초점) 높이 동일 유지.
DETECT_CHANNEL  = "auto"   # 검출 채널: "auto" | "gray" | "r" | "g" | "b" | "v"(RGB 최대값)
                           #  초록 바탕이면 "g" — 배경색과 같은 채널을 쓰면 바탕은 밝고
                           #  검은 마커만 어두워져 대비가 최대가 된다.
                           #  "auto"는 g → gray → v 순서로 시도해 성공한 채널을 사용.
BLUR_K          = 5        # 가우시안 블러 커널(홀수)
MIN_AREA_PX     = 1200     # 마커 후보 면적 범위(px^2)
MAX_AREA_PX     = 25000    # 구멍 창 전체가 후보로 잡히지 않게 마커 면적의 2~4배 수준 권장
ASPECT_RANGE    = (1.4, 2.8)   # 긴변/짧은변 허용 범위 (설계 2.0)
EXTENT_MIN      = 0.80     # 면적/외접사각형 면적 (직사각형이면 ~1)
ROI             = None     # 검출 영역 제한 (x, y, w, h). 예: (160, 100, 320, 280)
# --- 서브픽셀 에지 피팅 파라미터 ---
EDGE_EXCLUDE      = 0.20   # 각 변 양끝 제외 비율(코너 라운딩 회피)
PROFILE_HALF_PX   = 4.0    # 에지 법선 방향 프로파일 반길이(px)
N_PROFILE         = 17     # 프로파일 샘플 수
EDGE_CONTRAST_MIN = 10.0   # 프로파일 안/밖 밝기차 최소값(유색 바탕은 대비가 낮을 수 있음)
FIT_MIN_PTS       = 5      # 변당 직선 피팅 최소 포인트 수
REFINE_ITERS      = 2      # 에지 창 재중심화 반복 횟수
# ---------------------------------------------------------------------


def imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    """한글 경로(C:/Users/홍길동/...)에서도 안전한 이미지 저장."""
    ok, buf = cv2.imencode(path.suffix, img)
    if ok:
        buf.tofile(str(path))
    return bool(ok)


def imread_unicode(path: Path, flags=cv2.IMREAD_UNCHANGED):
    """한글 경로에서도 안전한 이미지 로드 (실패 시 None)."""
    try:
        data = np.fromfile(str(path), np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def load_calib(path: Path) -> dict:
    """캘리브 JSON 로드. 없거나 깨졌으면 빈 구조 반환."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("최상위가 객체가 아님")
    except FileNotFoundError:
        return {"version": 1}
    except (json.JSONDecodeError, ValueError, OSError) as e:
        print(f"[경고] 캘리브 파일을 읽을 수 없음({path}): {e} → 새로 만듦")
        return {"version": 1}
    data.setdefault("version", 1)
    return data


def save_calib(path: Path, data: dict) -> bool:
    """캘리브 JSON 저장(원자적 교체). 실패해도 촬영은 계속되도록 예외 삼킴."""
    data["version"] = 1
    data["updated"] = datetime.now().isoformat(timespec="seconds")
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(path)
        return True
    except OSError as e:
        print(f"[경고] 캘리브 파일 저장 실패({path}): {e}")
        return False


def zero_from_calib(data: dict):
    """캘리브 dict → 영점 참조. 필수 키가 없으면 None."""
    z = data.get("zero")
    if not isinstance(z, dict):
        return None
    try:
        ref = {"dx_px": float(z["dx_px"]), "dy_px": float(z["dy_px"]),
               "theta_deg": float(z["theta_deg"]),
               "scale_um": float(z["scale_um"]) if z.get("scale_um") else None}
    except (KeyError, TypeError, ValueError):
        print("[경고] 캘리브 파일의 zero 항목이 불완전함 → 영점 없음으로 시작")
        return None
    ref["time"] = z.get("time", "")
    ref["frame"] = z.get("frame", "")
    return ref


def zero_to_calib(data: dict, m, extra: dict = None) -> dict:
    """측정 dict → 캘리브 dict 의 zero 항목 갱신(메타 포함)."""
    d = m["det"]
    z = {"dx_px": round(m["dx_px"], 4), "dy_px": round(m["dy_px"], 4),
         "theta_deg": round(m["theta_deg"], 4),
         "scale_um": round(m["scale_um"], 4) if m["scale_um"] else None,
         "cx_px": round(d["cx"], 4), "cy_px": round(d["cy"], 4),
         "long_px": round(d["long_px"], 3),
         "time": datetime.now().isoformat(timespec="seconds")}
    if extra:
        z.update(extra)
    data["zero"] = z
    return data


def apply_calib_scale(data: dict) -> None:
    """캘리브 파일의 스케일을 전역 SCALE_UM_PER_PX 에 적용.

    파일 상단 상수를 직접 지정해 둔 경우엔 그 값이 우선(명시적 설정 존중).
    """
    global SCALE_UM_PER_PX
    s = data.get("scale")
    if not isinstance(s, dict) or not s.get("um_per_px"):
        return
    val = float(s["um_per_px"])
    if SCALE_UM_PER_PX is not None:
        if abs(SCALE_UM_PER_PX - val) > 1e-6:
            print(f"[스케일] 파일 상수({SCALE_UM_PER_PX} um/px)가 "
                  f"캘리브 파일 값({val})보다 우선 적용됨")
        return
    SCALE_UM_PER_PX = val
    print(f"[스케일] 캘리브 파일 값 적용: {val:.3f} um/px "
          f"(source={s.get('source', '?')})")


def print_calib(path: Path) -> None:
    """현재 캘리브 파일 내용 요약 출력."""
    if not path.exists():
        print(f"[캘리브] 파일 없음: {path}  (z 키로 영점을 잡으면 생성됨)")
        return
    data = load_calib(path)
    print(f"[캘리브] {path}   (updated: {data.get('updated', '-')})")
    s = data.get("scale")
    if isinstance(s, dict) and s.get("um_per_px"):
        print(f"  scale : {s['um_per_px']:.3f} um/px   source={s.get('source', '?')}"
              f"  pitch={s.get('pitch_mm', '-')}mm  퍼짐={s.get('spread_pct', '-')}%")
    else:
        print("  scale : 없음 (자동 스케일 = 마커 긴 변 기준 사용)")
    z = data.get("zero")
    if isinstance(z, dict):
        print(f"  zero  : dx0={z.get('dx_px')}px dy0={z.get('dy_px')}px "
              f"th0={z.get('theta_deg')}deg")
        print(f"          잡은 시각={z.get('time', '-')}  파일={z.get('frame', '-')}")
        print("  ※ 카메라·스테이지를 움직였다면 z 로 영점을 다시 잡을 것")
    else:
        print("  zero  : 없음 (z 키로 설정)")


def calc_scale_from_checkerboard(img_path: str, cb: str, pitch_mm: float,
                                 calib_path: Path = None) -> None:
    """체커보드 캡처(c 키 저장본)에서 μm/px 스케일 계산.

    cb: 내부 코너 수 "9x6" (칸 수 아님 — 10x7칸 보드의 내부 코너가 9x6).
    pitch_mm: 한 칸(정사각형)의 실치수 mm.
    출력된 SCALE_UM_PER_PX 값을 파일 상단 설정에 넣으면 자동 스케일보다 우선 적용된다.
    간격 퍼짐(min~max)이 크면 시야 내 배율 불균일(렌즈 왜곡) 신호이므로 함께 출력한다.
    """
    img = imread_unicode(Path(img_path))
    if img is None:
        sys.exit(f"[오류] 이미지를 읽을 수 없음: {img_path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint16:
        img = (img // 257).astype(np.uint8)
    try:
        cols, rows = (int(v) for v in cb.lower().split("x"))
    except ValueError:
        sys.exit("[오류] --cb 형식은 예: 9x6 (내부 코너 수)")
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(img, (cols, rows), flags=flags)
    if not ok:
        sys.exit("[오류] 체커보드 코너 검출 실패 — 초점/조명/코너 수(--cb)를 확인할 것")
    corners = cv2.cornerSubPix(
        img, corners, (5, 5), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3))
    pts = corners.reshape(rows, cols, 2)
    dh = np.linalg.norm(np.diff(pts, axis=1), axis=2)   # 가로 방향 인접 코너 간격
    dv = np.linalg.norm(np.diff(pts, axis=0), axis=2)   # 세로 방향 인접 코너 간격
    d_all = np.concatenate([dh.ravel(), dv.ravel()])
    px = float(np.median(d_all))
    scale = pitch_mm * 1000.0 / px
    spread_pct = 100.0 * (float(d_all.max()) - float(d_all.min())) / px
    print(f"[캘리브] 코너 {cols}x{rows} 검출 OK  ({img_path})")
    print(f"  칸 간격: median={px:.3f}px  mean={d_all.mean():.3f}px  std={d_all.std():.3f}px")
    print(f"  → SCALE_UM_PER_PX = {scale:.3f}   (칸 {pitch_mm}mm 기준)")
    print(f"  시야 배율 불균일 지표: 간격 min={d_all.min():.2f}px  max={d_all.max():.2f}px"
          f"  (퍼짐 {spread_pct:.1f}%)")
    print("  ↑ 퍼짐이 1~2%를 넘으면 렌즈 왜곡 보정(cv2.calibrateCamera + undistort) 고려.")
    if calib_path is not None:
        data = load_calib(calib_path)
        data["scale"] = {"um_per_px": round(scale, 4), "source": "checkerboard",
                         "pitch_mm": pitch_mm, "cb": cb,
                         "spread_pct": round(spread_pct, 2),
                         "image": str(img_path),
                         "time": datetime.now().isoformat(timespec="seconds")}
        if save_calib(calib_path, data):
            print(f"  → {calib_path} 에 저장됨. 다음 실행부터 자동 적용된다.")


def _v4l2_name(index: int) -> str:
    """Linux에서 /dev/video{index}의 장치명 조회 (현미경 식별용)."""
    try:
        return Path(f"/sys/class/video4linux/video{index}/name").read_text().strip()
    except OSError:
        return ""


def probe_cameras(max_index: int = 10) -> None:
    print(f"카메라 인덱스 탐색 중... [{CAP_NAME}] (카메라 점유 프로그램은 종료 상태여야 함)")
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, CAP_BACKEND)
        if cap.isOpened():
            ok, _ = cap.read()
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            name = "" if IS_WINDOWS else f"  ({_v4l2_name(i) or '/dev/video%d' % i})"
            print(f"  index {i}: {'프레임 OK' if ok else '열림(프레임 실패)'}  {w}x{h}{name}")
            if ok:
                found.append(i)
            cap.release()
    if not found:
        print("사용 가능한 카메라 없음. USB 연결과 점유 프로그램을 확인할 것.")
        if not IS_WINDOWS:
            print("  → 권한 확인: ls -l /dev/video*   (안 보이면 연결 문제)")
            print("  → 권한 부여: sudo usermod -aG video $USER  후 재로그인")
    else:
        print(f"→ 현미경으로 보이는 인덱스를 --index 로 지정 (후보: {found})")
        if not IS_WINDOWS:
            print("  (USB 카메라는 노드를 2개 만들므로 실제 영상 인덱스는 보통 짝수)")


def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, CAP_BACKEND)
    if not cap.isOpened():
        hint = ("HiView 등 점유 프로그램을 끄고, --probe로 인덱스를 확인할 것."
                if IS_WINDOWS else
                "점유 프로그램(cheese 등)을 끄고 --probe로 인덱스 확인. "
                "권한 오류면 sudo usermod -aG video $USER 후 재로그인.")
        sys.exit(f"[오류] 카메라 {index}를 열 수 없음. {hint}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC_STR))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)             # V4L2에서 유효(고인 프레임 감소)
    if EXPOSURE is not None:
        if IS_WINDOWS:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # DSHOW: 0.25=수동
            cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)    # log2초
        else:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)      # V4L2: 1=수동, 3=자동
            cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)    # 100μs 단위
    elif not IS_WINDOWS:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)          # V4L2 자동노출 명시
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)                # AWB 고정 시도(무시돼도 무해)

    for _ in range(30):                             # AE/AWB 안정화 워밍업 (~1s)
        cap.read()

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    v = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((v >> (8 * k)) & 0xFF) for k in range(4))
    print(f"[카메라] index={index} [{CAP_NAME}]  {w}x{h} @ {fps:.0f}fps  FOURCC={fourcc!r}")
    print(f"[노출]   auto={cap.get(cv2.CAP_PROP_AUTO_EXPOSURE):.2f}  "
          f"exposure={cap.get(cv2.CAP_PROP_EXPOSURE):.1f}  "
          f"(설정 요청: {'자동' if EXPOSURE is None else EXPOSURE})")
    if (w, h) != (WIDTH, HEIGHT):
        print(f"[경고] 요청 해상도와 다름({WIDTH}x{HEIGHT} 요청). 인덱스가 다른 카메라일 수 있음.")
    return cap


# ============================ 마커 검출/측정 ============================

def to_channel_f32(img: np.ndarray, ch: str = "gray") -> np.ndarray:
    """BGR(8/16bit) 또는 GRAY 입력 → float32 단일 채널(0..255).

    ch: "gray"(휘도) | "r" | "g" | "b" | "v"(RGB 최대값)
    초록 바탕 + 검은 마커라면 "g"가 대비 최대 (바탕 밝음, 마커 어두움).
    """
    if img.ndim == 2:
        g = img.astype(np.float32)
    elif ch == "gray":
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    elif ch == "v":
        g = img.max(axis=2).astype(np.float32)
    else:
        g = img[..., {"b": 0, "g": 1, "r": 2}[ch]].astype(np.float32)
    if img.dtype == np.uint16:
        g /= 257.0
    return g


def _channel_order():
    if DETECT_CHANNEL == "auto":
        return ["g", "gray", "v"]
    return [DETECT_CHANNEL]


def _bilinear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """부동소수 좌표 밝기 샘플링. 범위 밖은 NaN."""
    h, w = img.shape
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    valid = (x0 >= 0) & (y0 >= 0) & (x0 < w - 1) & (y0 < h - 1)
    x0c = np.clip(x0, 0, w - 2)
    y0c = np.clip(y0, 0, h - 2)
    fx = xs - x0c
    fy = ys - y0c
    v = (img[y0c, x0c] * (1 - fx) * (1 - fy) + img[y0c, x0c + 1] * fx * (1 - fy)
         + img[y0c + 1, x0c] * (1 - fx) * fy + img[y0c + 1, x0c + 1] * fx * fy)
    v = v.astype(np.float32)
    v[~valid] = np.nan
    return v


def _edge_cross(offsets: np.ndarray, vals: np.ndarray):
    """법선 프로파일(안:어두움 → 밖:밝음)에서 중간레벨 교차점(서브픽셀) 탐색."""
    if np.isnan(vals).any():
        return None
    v_in = float(np.percentile(vals, 12))    # 어두운 플래토(창 중심이 살짝 어긋나도 강건)
    v_out = float(np.percentile(vals, 88))   # 밝은 플래토
    if v_out - v_in < EDGE_CONTRAST_MIN:
        return None
    mid = 0.5 * (v_in + v_out)
    best, best_pos = None, None
    for k in range(len(vals) - 1):
        a, b = vals[k] - mid, vals[k + 1] - mid
        if a * b <= 0 and b != a:
            s = offsets[k] + (mid - vals[k]) * (offsets[k + 1] - offsets[k]) / (vals[k + 1] - vals[k])
            if best is None or abs(s) < abs(best_pos):
                best, best_pos = s, s
    return best


def _intersect(p1, d1, p2, d2):
    """두 직선(점+방향)의 교점."""
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], np.float64)
    b = np.array([p2[0] - p1[0], p2[1] - p1[1]], np.float64)
    det = np.linalg.det(A)
    if abs(det) < 1e-9:
        return None
    t = np.linalg.solve(A, b)
    return np.array([p1[0] + t[0] * d1[0], p1[1] + t[0] * d1[1]], np.float64)


def _fit_edges(gray: np.ndarray, corners: np.ndarray, center: np.ndarray,
               offsets: np.ndarray):
    """현재 코너 추정으로 정의된 네 변 각각에 서브픽셀 직선 피팅."""
    lines, all_res = [], []
    for i in range(4):
        p0, p1 = corners[i], corners[(i + 1) % 4]
        L = float(np.linalg.norm(p1 - p0))
        if L < 4:
            lines.append(None)
            continue
        dirv = (p1 - p0) / L
        nrm = np.array([-dirv[1], dirv[0]])
        midp = 0.5 * (p0 + p1)
        if np.dot(nrm, midp - center) < 0:                     # 법선을 바깥(밝은 쪽)으로
            nrm = -nrm
        K = max(FIT_MIN_PTS + 2, int(L * (1 - 2 * EDGE_EXCLUDE) / 3))
        ts = np.linspace(EDGE_EXCLUDE, 1 - EDGE_EXCLUDE, K)
        pts = []
        for t in ts:
            base = p0 + t * (p1 - p0)
            xs = base[0] + offsets * nrm[0]
            ys = base[1] + offsets * nrm[1]
            s = _edge_cross(offsets, _bilinear(gray, xs, ys))
            if s is not None:
                pts.append(base + s * nrm)
        if len(pts) < FIT_MIN_PTS:
            lines.append(None)
            continue
        P = np.array(pts, np.float32)
        vx, vy, x0, y0 = cv2.fitLine(P, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        d = np.array([vx, vy], np.float64)
        p = np.array([x0, y0], np.float64)
        res = np.abs((P[:, 0] - p[0]) * d[1] - (P[:, 1] - p[1]) * d[0])
        all_res.extend(res.tolist())
        lines.append((p, d))
    return lines, all_res


def _corners_from_lines(lines):
    """인접한 네 직선의 교점 → 코너 4개."""
    corners = []
    for i in range(4):
        q = _intersect(lines[i][0], lines[i][1],
                       lines[(i + 1) % 4][0], lines[(i + 1) % 4][1])
        if q is None:
            return None
        corners.append(q)
    return np.array(corners)


def _pick_candidate(g8: np.ndarray, thr: float):
    """주어진 임계값으로 이진화 → 필터 통과 후보 중 설계 종횡비(2:1)에
    가장 가까운 minAreaRect 반환. 없으면 None."""
    _, bw = cv2.threshold(g8, thr, 255, cv2.THRESH_BINARY_INV)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    if ROI is not None:
        m = np.zeros_like(bw)
        x, y, w, h = ROI
        m[y:y + h, x:x + w] = 255
        bw = cv2.bitwise_and(bw, m)

    expected = (MARKER_LONG_MM / MARKER_SHORT_MM) if (MARKER_LONG_MM and MARKER_SHORT_MM) else 2.0
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best = None
    for c in cnts:
        a = cv2.contourArea(c)
        if not (MIN_AREA_PX <= a <= MAX_AREA_PX):
            continue
        rect = cv2.minAreaRect(c)
        (rcx, rcy), (rw, rh), _ = rect
        short, long_ = sorted([rw, rh])
        if short <= 1:
            continue
        aspect = long_ / short
        if not (ASPECT_RANGE[0] <= aspect <= ASPECT_RANGE[1]):
            continue
        if a / (rw * rh) < EXTENT_MIN:
            continue
        score = abs(aspect - expected)
        if best is None or score < best[0]:
            best = (score, rect)
    return best[1] if best else None


def detect_rect(gray: np.ndarray):
    """검은 8x4 직사각형 마커 검출 → 서브픽셀 포즈.

    반환: dict(cx, cy, theta_deg, long_px, short_px, rms, corners, fallback) 또는 None
      cx, cy    : 마커 중심(px, 서브픽셀)
      theta_deg : 긴 축 각도(도), (-90, +90]
      rms       : 에지 직선 피팅 잔차 RMS(px) — 품질 지표
      fallback  : True면 서브픽셀 실패로 minAreaRect 값 사용
    """
    g = cv2.GaussianBlur(gray, (BLUR_K, BLUR_K), 0)
    g8 = np.clip(g, 0, 255).astype(np.uint8)

    # 다단계 임계값: 전역 Otsu → 실패 시 어두운 쪽/밝은 쪽 계층을 재분할
    # (플레이트 밑면-초록 바탕-검은 마커처럼 3개 밝기 계층이 섞인 장면 대응)
    thr1, _ = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rect, stage = _pick_candidate(g8, thr1), 1
    if rect is None:
        dark = g8[g8 <= thr1]
        if dark.size > 500:
            thr2, _ = cv2.threshold(dark.reshape(-1, 1), 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            rect, stage = _pick_candidate(g8, thr2), 2
    if rect is None:
        bright = g8[g8 > thr1]
        if bright.size > 500:
            thr3, _ = cv2.threshold(bright.reshape(-1, 1), 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            rect, stage = _pick_candidate(g8, thr3), 3
    if rect is None:
        return None
    corners0 = cv2.boxPoints(rect).astype(np.float64)          # 4x2, 순차
    center0 = np.array(rect[0], np.float64)

    # --- 네 변: 중앙부 샘플 → 법선 프로파일 → 서브픽셀 에지점 → 직선 피팅.
    #     피팅된 변으로 창을 재중심화하며 REFINE_ITERS회 반복 정련 ---
    offsets = np.linspace(-PROFILE_HALF_PX, PROFILE_HALF_PX, N_PROFILE)
    corners_cur, center_cur = corners0, center0
    lines, all_res, ok_sub = None, [], True
    for _ in range(REFINE_ITERS):
        lines, all_res = _fit_edges(gray, corners_cur, center_cur, offsets)
        if any(l is None for l in lines):
            ok_sub = False
            break
        c = _corners_from_lines(lines)
        if c is None:
            ok_sub = False
            break
        corners_cur = c
        center_cur = c.mean(axis=0)

    if not ok_sub:
        # 서브픽셀 실패 → minAreaRect 폴백
        (rcx, rcy), (rw, rh), ang = rect
        if rw < rh:
            rw, rh = rh, rw
            ang += 90.0
        while ang <= -90:
            ang += 180
        while ang > 90:
            ang -= 180
        return {"cx": float(rcx), "cy": float(rcy), "theta_deg": float(ang),
                "long_px": float(rw), "short_px": float(rh), "rms": None,
                "corners": corners0, "fallback": True, "stage": stage}

    corners = corners_cur
    center = center_cur

    elens = [float(np.linalg.norm(corners[(i + 1) % 4] - corners[i])) for i in range(4)]
    if elens[0] + elens[2] >= elens[1] + elens[3]:
        li, lj = 0, 2
    else:
        li, lj = 1, 3
    d1 = corners[(li + 1) % 4] - corners[li]
    d2 = corners[(lj + 1) % 4] - corners[lj]
    if np.dot(d1, d2) < 0:
        d2 = -d2
    dlong = d1 / np.linalg.norm(d1) + d2 / np.linalg.norm(d2)
    theta = float(np.degrees(np.arctan2(dlong[1], dlong[0])))
    while theta <= -90:
        theta += 180
    while theta > 90:
        theta -= 180
    long_px = 0.5 * (elens[li] + elens[lj])
    short_px = 0.5 * (elens[(li + 1) % 4] + elens[(lj + 1) % 4])
    rms = float(np.sqrt(np.mean(np.square(all_res)))) if all_res else None
    return {"cx": float(center[0]), "cy": float(center[1]), "theta_deg": theta,
            "long_px": float(long_px), "short_px": float(short_px), "rms": rms,
            "corners": corners, "fallback": False, "stage": stage}


def measure_offset(img: np.ndarray):
    """마커 검출 후 '이미지 중심' 대비 오프셋 + 회전 계산."""
    det, shape = None, None
    for ch in _channel_order():
        g = to_channel_f32(img, ch)
        det = detect_rect(g)
        if det is not None:
            det["channel"] = ch
            shape = g.shape
            break
    if det is None:
        return None
    h, w = shape
    dx = det["cx"] - (w - 1) / 2.0
    dy = det["cy"] - (h - 1) / 2.0
    out = {"det": det, "dx_px": dx, "dy_px": dy,
           "r_off_px": float(np.hypot(dx, dy)), "theta_deg": det["theta_deg"],
           "scale_um": None, "dx_mm": None, "dy_mm": None, "r_off_mm": None}
    scale = SCALE_UM_PER_PX
    if scale is None and MARKER_LONG_MM and det["long_px"] > 0:
        scale = MARKER_LONG_MM * 1000.0 / det["long_px"]       # 긴 변 8mm 기준 자동 스케일
    if scale:
        out["scale_um"] = float(scale)
        out["dx_mm"] = dx * scale / 1000.0
        out["dy_mm"] = dy * scale / 1000.0
        out["r_off_mm"] = out["r_off_px"] * scale / 1000.0
    return out


def fmt_offset(m) -> str:
    if m is None:
        return "마커 미검출"
    s = (f"dx={m['dx_px']:+7.2f}px dy={m['dy_px']:+7.2f}px "
         f"|r|={m['r_off_px']:6.2f}px  theta={m['theta_deg']:+6.2f}deg")
    if m["dx_mm"] is not None:
        s += (f"   ({m['dx_mm']:+.3f}, {m['dy_mm']:+.3f}) mm @ {m['scale_um']:.1f}um/px")
    d = m["det"]
    if d.get("fallback"):
        s += "  [폴백: 서브픽셀 실패]"
    elif d.get("rms") is not None:
        s += f"  rms={d['rms']:.2f}px"
    s += f"  ch={d.get('channel', '-')}/t{d.get('stage', '-')}"
    return s


def rel_to_zero(m, zero_ref):
    """현재 측정 m 을 영점(zero_ref) 대비 상대 포즈로 변환.

    px 단위로 벡터 차를 구한 뒤 현재 프레임 스케일로 mm 환산한다.
    (자동 스케일 사용 시 영점 촬영과 현재 프레임의 z 높이가 같아야 정확)
    변위 검증·본실험 분석에는 반드시 이 Δ값을 쓴다 — 시작 위치가 이미지
    중심에서 벗어나 있어도 그 오프셋이 상쇄된다.
    """
    if m is None or zero_ref is None:
        return None
    ddx = m["dx_px"] - zero_ref["dx_px"]
    ddy = m["dy_px"] - zero_ref["dy_px"]
    dth = m["theta_deg"] - zero_ref["theta_deg"]
    while dth <= -90:
        dth += 180
    while dth > 90:
        dth -= 180
    out = {"ddx_px": float(ddx), "ddy_px": float(ddy),
           "dr_px": float(np.hypot(ddx, ddy)), "dtheta_deg": float(dth),
           "ddx_mm": None, "ddy_mm": None, "dr_mm": None}
    scale = m.get("scale_um") or zero_ref.get("scale_um")
    if scale:
        out["ddx_mm"] = out["ddx_px"] * scale / 1000.0
        out["ddy_mm"] = out["ddy_px"] * scale / 1000.0
        out["dr_mm"] = out["dr_px"] * scale / 1000.0
    return out


def fmt_rel(rel) -> str:
    if rel is None:
        return "영점 미설정"
    s = (f"Δdx={rel['ddx_px']:+7.2f}px Δdy={rel['ddy_px']:+7.2f}px "
         f"|Δr|={rel['dr_px']:6.2f}px  Δθ={rel['dtheta_deg']:+6.2f}deg")
    if rel["dr_mm"] is not None:
        s += (f"   (Δ {rel['ddx_mm']:+.3f}, {rel['ddy_mm']:+.3f}) mm"
              f"  |Δr|={rel['dr_mm']:.3f}mm")
    return s


def draw_detection(disp: np.ndarray, m, zero_ref=None) -> np.ndarray:
    """오버레이: 중심 십자 + 검출 사각형 + 긴축 표시 + 수치 (+영점 대비 Δ)."""
    h, w = disp.shape[:2]
    cx0, cy0 = int((w - 1) / 2), int((h - 1) / 2)
    cv2.line(disp, (cx0, 0), (cx0, h), (0, 255, 0), 1)
    cv2.line(disp, (0, cy0), (w, cy0), (0, 255, 0), 1)
    if zero_ref is not None:                       # 영점 위치 표시(마젠타 X)
        zx = int(round(cx0 + zero_ref["dx_px"]))
        zy = int(round(cy0 + zero_ref["dy_px"]))
        cv2.drawMarker(disp, (zx, zy), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 1)
    if m is None:
        cv2.putText(disp, "NO MARKER", (8, 44), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return disp
    d = m["det"]
    pts = np.round(d["corners"]).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(disp, [pts], True, (0, 0, 255), 1, cv2.LINE_AA)
    cx, cy = int(round(d["cx"])), int(round(d["cy"]))
    cv2.drawMarker(disp, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 12, 1)
    th = np.radians(d["theta_deg"])
    ax = (int(round(cx + 0.6 * d["long_px"] * np.cos(th))),
          int(round(cy + 0.6 * d["long_px"] * np.sin(th))))
    cv2.line(disp, (cx, cy), ax, (255, 200, 0), 1, cv2.LINE_AA)
    cv2.line(disp, (cx0, cy0), (cx, cy), (0, 255, 255), 1, cv2.LINE_AA)
    txt = (f"dx={m['dx_px']:+.1f} dy={m['dy_px']:+.1f} th={m['theta_deg']:+.2f}deg")
    if m["r_off_mm"] is not None:
        txt += f" ({m['r_off_mm']:.3f}mm)"
    cv2.putText(disp, txt, (8, 44), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 255), 1, cv2.LINE_AA)
    rel = rel_to_zero(m, zero_ref)
    if rel is not None:                            # 영점→마커 벡터 + Δ 수치(마젠타)
        zx = int(round(cx0 + zero_ref["dx_px"]))
        zy = int(round(cy0 + zero_ref["dy_px"]))
        cv2.line(disp, (zx, zy), (cx, cy), (255, 0, 255), 1, cv2.LINE_AA)
        if rel["dr_mm"] is not None:
            t2 = (f"Z: dx={rel['ddx_mm']:+.3f} dy={rel['ddy_mm']:+.3f}"
                  f" r={rel['dr_mm']:.3f}mm dth={rel['dtheta_deg']:+.2f}deg")
        else:
            t2 = (f"Z: dx={rel['ddx_px']:+.1f} dy={rel['ddy_px']:+.1f}px"
                  f" r={rel['dr_px']:.1f}px dth={rel['dtheta_deg']:+.2f}deg")
        cv2.putText(disp, t2, (8, 66), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 0, 255), 1, cv2.LINE_AA)
    return disp


# ============================ 촬영/저장 ============================

def capture_burst(cap: cv2.VideoCapture, n: int = N_BURST):
    """버퍼 플러시 → n장 버스트 → 밝기 이상 프레임(USB 글리치) 제외 → 평균."""
    for _ in range(N_FLUSH):
        cap.grab()

    frames = []
    for _ in range(n):
        ok, f = cap.read()
        if ok and f is not None:
            frames.append(f)
    if len(frames) < max(3, n // 2 + 1):
        return None, None, 0

    stack = np.stack(frames).astype(np.float32)
    means = stack.mean(axis=(1, 2, 3))
    med = float(np.median(means))
    mad = float(np.median(np.abs(means - med)))
    keep = np.abs(means - med) <= (6.0 * mad + 1.0)
    if not keep.any():
        keep[:] = True
    used = int(keep.sum())

    avg16 = np.round(stack[keep].mean(axis=0) * 257.0).astype(np.uint16)
    return avg16, frames[0], used


def capture_and_save(cap, outdir: Path, label: str, counters: dict, logw,
                     zero_ref=None):
    """한 번의 트리거: 버스트 촬영 → 마커 측정 → 저장 → CSV 로그.

    zero_ref 가 주어지면 영점 대비 Δ(d_dx_mm 등)도 함께 출력·기록한다.
    반환: (저장 파일 리스트, 측정 dict 또는 None)
    """
    counters[label] = counters.get(label, 0) + 1
    n = counters[label]
    ts = datetime.now().strftime("%H%M%S")

    print(f"[촬영] {label} #{n:03d} ... ", end="", flush=True)
    avg16, raw, used = capture_burst(cap)
    if avg16 is None:
        counters[label] -= 1
        print("실패 (프레임 수신 불량)")
        return [], None

    m = measure_offset(avg16)
    rel = rel_to_zero(m, zero_ref)

    saved = []
    p_avg = outdir / f"{label}_{n:03d}_{ts}.png"
    if imwrite_unicode(p_avg, avg16):
        saved.append(p_avg)
    if SAVE_RAW and raw is not None:
        p_raw = outdir / f"{label}_{n:03d}_{ts}_raw.png"
        if imwrite_unicode(p_raw, raw):
            saved.append(p_raw)
    if SAVE_ANNOTATED:
        disp = (avg16 // 257).astype(np.uint8)
        disp = draw_detection(disp, m, zero_ref)
        p_det = outdir / f"{label}_{n:03d}_{ts}_det.png"
        if imwrite_unicode(p_det, disp):
            saved.append(p_det)

    row = [datetime.now().isoformat(timespec="seconds"), label, n,
           saved[0].name if saved else "", used, N_BURST]
    if m is not None:
        d = m["det"]
        row += [f"{d['cx']:.3f}", f"{d['cy']:.3f}", f"{m['theta_deg']:.3f}",
                f"{m['dx_px']:.3f}", f"{m['dy_px']:.3f}", f"{m['r_off_px']:.3f}",
                f"{d['long_px']:.2f}", f"{d['short_px']:.2f}",
                f"{d['rms']:.3f}" if d["rms"] is not None else "",
                "1" if d.get("fallback") else "0",
                d.get("channel", ""), str(d.get("stage", "")),
                f"{m['scale_um']:.2f}" if m["scale_um"] else "",
                f"{m['dx_mm']:.4f}" if m["dx_mm"] is not None else "",
                f"{m['dy_mm']:.4f}" if m["dy_mm"] is not None else "",
                f"{m['r_off_mm']:.4f}" if m["r_off_mm"] is not None else ""]
    else:
        row += [""] * 16
    if rel is not None and rel["dr_mm"] is not None:
        row += [f"{rel['ddx_mm']:.4f}", f"{rel['ddy_mm']:.4f}",
                f"{rel['dr_mm']:.4f}", f"{rel['dtheta_deg']:.3f}"]
    else:
        row += [""] * 4
    logw.writerow(row)

    print(f"저장 ({used}/{N_BURST} 평균) → {saved[0].name}")
    print(f"[측정] {fmt_offset(m)}")
    if rel is not None:
        print(f"[영점대비] {fmt_rel(rel)}")
    return saved, m


def make_session_dir() -> Path:
    outdir = OUT_ROOT / datetime.now().strftime("%Y-%m-%d_%H%M")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def open_log(outdir: Path):
    f = open(outdir / "log.csv", "a", newline="", encoding="utf-8-sig")
    w = csv.writer(f)
    if f.tell() == 0:
        w.writerow(["time", "label", "trial", "file", "frames_used", "frames_requested",
                    "cx_px", "cy_px", "theta_deg", "dx_px", "dy_px", "r_off_px",
                    "long_px", "short_px", "fit_rms_px", "fallback", "channel", "thr_stage",
                    "scale_um_per_px", "dx_mm", "dy_mm", "r_off_mm",
                    "d_dx_mm", "d_dy_mm", "d_r_mm", "d_theta_deg"])
    return f, w


def zero_ref_from(m):
    """측정 dict → 영점 참조(dict). 마커 미검출이면 None."""
    if m is None:
        return None
    return {"dx_px": m["dx_px"], "dy_px": m["dy_px"],
            "theta_deg": m["theta_deg"], "scale_um": m["scale_um"]}


# ============================ 실행 모드 ============================

def main_interactive(cap, outdir: Path, calib_path: Path, ignore_zero: bool = False):
    logf, logw = open_log(outdir)
    mode = "rigid"
    counters = {}
    calib = load_calib(calib_path)
    zero_ref = None if ignore_zero else zero_from_calib(calib)   # 파일에 저장된 영점 재사용
    prev_zero = None                                             # d(취소) 시 복원용
    last_saved, last_label = [], ""
    print(f"[세션] {outdir}   (SPACE 촬영 / r,s 모드 / z 영점 / c 캘리브 / d 취소 / q 종료)")
    if zero_ref is not None:
        print(f"[영점] {calib_path} 에서 불러옴: dx0={zero_ref['dx_px']:+.2f}px "
              f"dy0={zero_ref['dy_px']:+.2f}px th0={zero_ref['theta_deg']:+.2f}deg "
              f"(잡은 시각 {zero_ref.get('time', '-')})")
        print("       ※ 카메라를 옮겼거나 초점을 바꿨다면 z 로 다시 잡을 것")
    elif ignore_zero:
        print("[영점] --ignore-zero: 저장된 영점 무시. z 로 새로 잡을 것")
    else:
        print("       변위 측정 절차: 마커 시작 위치에서 z → 이동 → SPACE → 화면 'Z:' 값 사용")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            m_live = measure_offset(frame)          # 프리뷰 실시간 측정
            disp = draw_detection(frame.copy(), m_live, zero_ref)
            nxt = counters.get(mode, 0) + 1
            zt = "ZERO OK" if zero_ref is not None else "NO ZERO"
            cv2.putText(disp, f"MODE: {mode}  next trial: {nxt:03d}  [{zt}]", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            if last_saved:
                cv2.putText(disp, f"last: {last_saved[0].name}",
                            (8, disp.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            try:
                cv2.imshow(WINDOW, disp)
            except cv2.error:
                sys.exit("[오류] GUI 창을 열 수 없음(헤드리스/SSH 환경 또는 headless OpenCV). "
                         "--interval 모드를 사용하거나 데스크톱 세션에서 실행할 것.")
            k = cv2.waitKey(1) & 0xFF
            if 65 <= k <= 90:                       # Caps Lock 대문자 → 소문자 처리
                k += 32

            if k == ord("q"):
                break
            elif k == ord("r"):
                mode = "rigid"
            elif k == ord("s"):
                mode = "soft"
            elif k == 32:                                   # SPACE
                last_saved, _m = capture_and_save(cap, outdir, mode, counters, logw,
                                                  zero_ref)
                last_label = mode
                logf.flush()
            elif k == ord("z"):
                last_saved, mz = capture_and_save(cap, outdir, "zero", counters, logw)
                last_label = "zero"
                new_zero = zero_ref_from(mz)
                if new_zero is not None:
                    prev_zero = zero_ref
                    zero_ref = new_zero
                    zero_to_calib(calib, mz,
                                  {"frame": last_saved[0].name if last_saved else ""})
                    save_calib(calib_path, calib)
                    print(f"[영점] 설정·저장: dx0={mz['dx_px']:+.2f}px dy0={mz['dy_px']:+.2f}px "
                          f"th0={mz['theta_deg']:+.2f}deg → {calib_path}")
                else:
                    print("[영점] 마커 미검출 → 영점 갱신 안 함"
                          + ("" if zero_ref is None else " (기존 영점 유지)"))
                logf.flush()
            elif k == ord("c"):
                last_saved, _m = capture_and_save(cap, outdir, "calib", counters, logw)
                last_label = "calib"
                logf.flush()
            elif k == ord("d") and last_saved:
                for p in last_saved:
                    Path(p).unlink(missing_ok=True)
                counters[last_label] = max(0, counters.get(last_label, 1) - 1)
                logw.writerow([datetime.now().isoformat(timespec="seconds"),
                               "deleted", "", last_saved[0].name] + [""] * 22)
                logf.flush()
                print(f"[취소] {last_saved[0].name} 삭제, 번호 되돌림")
                if last_label == "zero":
                    zero_ref = prev_zero
                    if zero_ref is None:
                        calib.pop("zero", None)
                        print("[취소] 영점도 해제됨 — z 로 다시 설정할 것")
                    else:
                        calib["zero"] = {
                            "dx_px": round(zero_ref["dx_px"], 4),
                            "dy_px": round(zero_ref["dy_px"], 4),
                            "theta_deg": round(zero_ref["theta_deg"], 4),
                            "scale_um": zero_ref.get("scale_um"),
                            "time": zero_ref.get("time", ""),
                            "frame": zero_ref.get("frame", "")}
                        print(f"[취소] 직전 영점으로 복원 "
                              f"(dx0={zero_ref['dx_px']:+.2f}px dy0={zero_ref['dy_px']:+.2f}px)")
                    save_calib(calib_path, calib)
                    prev_zero = None
                last_saved = []
    finally:
        logf.close()
        cap.release()
        cv2.destroyAllWindows()


def main_interval(cap, outdir: Path, interval: float, count: int, label: str,
                  calib_path: Path = None, ignore_zero: bool = False):
    """무인 자동 촬영·측정. 예: --interval 20 --count 100 (노이즈 플로어/드리프트 테스트).

    캘리브 파일에 영점이 있으면 그 영점 기준으로 Δ가 기록된다.
    없으면 첫 성공 측정을 세션 한정 기준으로 삼는다(파일에는 쓰지 않음).
    """
    logf, logw = open_log(outdir)
    print(f"[세션] {outdir}  {interval}s 간격 {count}장, label={label} (Ctrl+C 중단)")
    counters = {}
    zero_ref = None
    if calib_path is not None and not ignore_zero:
        zero_ref = zero_from_calib(load_calib(calib_path))
        if zero_ref is not None:
            print(f"[영점] {calib_path} 에서 불러옴: dx0={zero_ref['dx_px']:+.2f}px "
                  f"dy0={zero_ref['dy_px']:+.2f}px")
    try:
        for i in range(count):
            t0 = time.time()
            _saved, m = capture_and_save(cap, outdir, label, counters, logw, zero_ref)
            if zero_ref is None and m is not None:
                zero_ref = zero_ref_from(m)
                print("    [영점] 저장된 영점이 없어 첫 측정을 세션 기준으로 사용(파일 미저장)")
            logf.flush()
            print(f"    진행 {i + 1}/{count}")
            while time.time() - t0 < interval:              # 대기 중에도 프레임 소비
                cap.read()                                  # (버퍼/AE 신선 유지)
                time.sleep(0.005)
    except KeyboardInterrupt:
        print("\n[중단] 사용자 요청")
    finally:
        logf.close()
        cap.release()


def parse_args():
    ap = argparse.ArgumentParser(description="USB 현미경 자동 촬영 + 직사각형 마커 포즈 측정")
    ap.add_argument("--probe", action="store_true", help="카메라 인덱스 탐색 후 종료")
    ap.add_argument("--index", type=int, default=0, help="카메라 인덱스 (기본 0)")
    ap.add_argument("--interval", type=float, default=None, help="자동 촬영 간격(초)")
    ap.add_argument("--count", type=int, default=100, help="자동 촬영 장수 (기본 100)")
    ap.add_argument("--label", type=str, default="drift", help="자동 촬영 파일 라벨")
    ap.add_argument("--calc-scale", type=str, default=None, metavar="IMG",
                    help="체커보드 캡처 이미지에서 μm/px 계산 후 종료 (--cb, --pitch 함께)")
    ap.add_argument("--cb", type=str, default="9x6",
                    help="체커보드 내부 코너 수 CxR (기본 9x6)")
    ap.add_argument("--pitch", type=float, default=1.0,
                    help="체커보드 한 칸 크기 mm (기본 1.0)")
    ap.add_argument("--calib", type=str, default=str(CALIB_FILE),
                    help=f"영점·스케일 캘리브 JSON 경로 (기본 {CALIB_FILE})")
    ap.add_argument("--show-calib", action="store_true",
                    help="현재 캘리브 파일 내용 출력 후 종료")
    ap.add_argument("--clear-zero", action="store_true",
                    help="캘리브 파일의 영점 삭제 후 종료")
    ap.add_argument("--ignore-zero", action="store_true",
                    help="이번 실행에만 저장된 영점을 쓰지 않음(파일은 그대로)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    calib_path = Path(args.calib)
    if args.probe:
        probe_cameras()
        sys.exit(0)
    if args.show_calib:
        print_calib(calib_path)
        sys.exit(0)
    if args.clear_zero:
        data = load_calib(calib_path)
        if data.pop("zero", None) is None:
            print(f"[캘리브] 지울 영점이 없음: {calib_path}")
        elif save_calib(calib_path, data):
            print(f"[캘리브] 영점 삭제됨: {calib_path}  (스케일은 유지)")
        sys.exit(0)
    if args.calc_scale:
        calc_scale_from_checkerboard(args.calc_scale, args.cb, args.pitch, calib_path)
        sys.exit(0)

    apply_calib_scale(load_calib(calib_path))       # 저장된 스케일 자동 적용
    cam = open_camera(args.index)
    session = make_session_dir()
    if args.interval is not None:
        main_interval(cam, session, args.interval, args.count, args.label,
                      calib_path, args.ignore_zero)
    else:
        main_interactive(cam, session, calib_path, args.ignore_zero)