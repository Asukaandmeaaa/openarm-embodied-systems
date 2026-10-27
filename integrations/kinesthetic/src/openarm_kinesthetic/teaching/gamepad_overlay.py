"""Small in-viewer gamepad input overlay rendered with Pillow."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class GamepadOverlay:
    """Draw a compact controller whose active inputs light up."""

    WIDTH = 420
    HEIGHT = 230
    BACKGROUND = (9, 15, 23)
    PANEL = (8, 83, 145)
    OUTLINE = (41, 193, 255)
    MUTED = (62, 78, 98)
    TEXT = (229, 235, 242)
    ACTIVE = (49, 220, 112)
    MOTION = (64, 170, 255)
    STOP = (255, 76, 88)
    YELLOW = (255, 213, 64)

    def __init__(self, config: dict[str, Any], *, highlight_hold_s: float = 0.65):
        self.axis = {name: int(index) for name, index in config["mapping"]["axes"].items()}
        self.button = {name: int(index) for name, index in config["mapping"]["buttons"].items()}
        control = config["control"]
        self.deadzone = float(control["stick_deadzone"])
        self.trigger_threshold = float(control["trigger_press_threshold"])
        self.highlight_hold_s = float(highlight_hold_s)
        self.last_control = "NONE"
        self.last_control_at_s = -1e9
        try:
            self.font = ImageFont.truetype("DejaVuSans.ttf", 11)
            self.font_bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 11)
        except OSError:  # pragma: no cover - system font fallback
            self.font = ImageFont.load_default()
            self.font_bold = self.font

    @staticmethod
    def _raw_axis(axes: np.ndarray, index: int) -> float:
        if index >= len(axes):
            return 0.0
        return float(np.clip(float(axes[index]) / 32767.0, -1.0, 1.0))

    def _axis(self, axes: np.ndarray, name: str) -> float:
        return self._raw_axis(axes, self.axis[name])

    def _trigger(self, axes: np.ndarray, name: str) -> bool:
        return 0.5 * (self._axis(axes, name) + 1.0) >= self.trigger_threshold

    def _button(self, buttons: np.ndarray, name: str) -> bool:
        index = self.button[name]
        return bool(buttons[index]) if index < len(buttons) else False

    @staticmethod
    def _circle(draw: ImageDraw.ImageDraw, xy: tuple[int, int], radius: int, **kwargs) -> None:
        x, y = xy
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), **kwargs)

    def _centered_text(
        self,
        draw: ImageDraw.ImageDraw,
        xy: tuple[int, int],
        label: str,
        *,
        fill: tuple[int, int, int],
        bold: bool = True,
    ) -> None:
        font = self.font_bold if bold else self.font
        bounds = draw.textbbox((0, 0), label, font=font)
        draw.text(
            (xy[0] - (bounds[2] - bounds[0]) / 2, xy[1] - (bounds[3] - bounds[1]) / 2 - 1),
            label,
            fill=fill,
            font=font,
        )

    @staticmethod
    def _cubic(
        start: tuple[float, float],
        control1: tuple[float, float],
        control2: tuple[float, float],
        end: tuple[float, float],
        count: int = 12,
    ) -> list[tuple[int, int]]:
        points=[]
        for index in range(1,count+1):
            t=index/count;u=1.0-t
            x=u**3*start[0]+3*u*u*t*control1[0]+3*u*t*t*control2[0]+t**3*end[0]
            y=u**3*start[1]+3*u*u*t*control1[1]+3*u*t*t*control2[1]+t**3*end[1]
            points.append((round(x),round(y)))
        return points

    @classmethod
    def _body_outline(cls) -> list[tuple[int, int]]:
        points=[(55,61)]
        segments=(
            ((55,61),(105,43),(315,43),(365,61)),
            ((365,61),(394,71),(409,103),(405,137)),
            ((405,137),(403,162),(394,197),(382,216)),
            ((382,216),(371,229),(349,225),(337,211)),
            ((337,211),(321,192),(307,174),(289,164)),
            ((289,164),(271,154),(253,170),(242,180)),
            ((242,180),(228,193),(192,193),(178,180)),
            ((178,180),(167,170),(149,154),(131,164)),
            ((131,164),(113,174),(99,192),(83,211)),
            ((83,211),(71,225),(49,229),(38,216)),
            ((38,216),(26,197),(17,162),(15,137)),
            ((15,137),(11,103),(26,71),(55,61)),
        )
        for segment in segments:
            points.extend(cls._cubic(*segment))
        return points

    def _current_control(
        self,
        *,
        lt: bool,
        rt: bool,
        rb: bool,
        a: bool,
        y: bool,
        lx: float,
        ly: float,
        rx: float,
        ry: float,
        dx: float,
        dy: float,
    ) -> str | None:
        if y:
            return "Y / CLOSE"
        if a:
            return "A / OPEN"
        if rb:
            return "RB / STOP"
        if rt:
            return "RT / STOP"
        if abs(dx) > 0.5 or abs(dy) > 0.5:
            return "D-PAD / Z+ROLL"
        if max(abs(rx), abs(ry)) > self.deadzone:
            return "RIGHT STICK / POSE"
        if max(abs(lx), abs(ly)) > self.deadzone:
            return "LEFT STICK / X-Y"
        if lt:
            return "LT / ENABLE"
        return None

    def render(
        self,
        axes: np.ndarray,
        buttons: np.ndarray,
        *,
        connected: bool,
        active: bool,
        stop_latched: bool,
        now_s: float,
    ) -> np.ndarray:
        axes = np.asarray(axes)
        buttons = np.asarray(buttons)
        lx=self._axis(axes,"left_x");ly=self._axis(axes,"left_y")
        rx=self._axis(axes,"right_x");ry=self._axis(axes,"right_y")
        dx=self._axis(axes,"dpad_x");dy=self._axis(axes,"dpad_y")
        lt=self._trigger(axes,"lt") if connected else False
        rt=self._trigger(axes,"rt") if connected else False
        a=self._button(buttons,"gripper_open_a") if connected else False
        y=self._button(buttons,"gripper_close_y") if connected else False
        rb=self._button(buttons,"stop_rb_r1") if connected else False
        current=self._current_control(lt=lt,rt=rt,rb=rb,a=a,y=y,lx=lx,ly=ly,rx=rx,ry=ry,dx=dx,dy=dy)
        if current is not None:
            self.last_control=current;self.last_control_at_s=float(now_s)
        shown=self.last_control if float(now_s)-self.last_control_at_s<=self.highlight_hold_s else "NONE"

        image=Image.new("RGB",(self.WIDTH,self.HEIGHT),self.BACKGROUND)
        draw=ImageDraw.Draw(image)
        draw.rounded_rectangle((2,2,self.WIDTH-3,self.HEIGHT-3),radius=15,fill=self.BACKGROUND,outline=(36,52,68),width=2)
        status="ACTIVE" if active else "STOP" if stop_latched else "READY" if connected else "DISCONNECTED"
        status_color=self.ACTIVE if active else self.STOP if stop_latched or not connected else self.TEXT
        draw.text((14,9),f"G3V2  {status}",fill=status_color,font=self.font_bold)
        draw.text((190,9),f"LAST  {shown}",fill=self.TEXT,font=self.font)

        body=self._body_outline()
        body_outline=self.ACTIVE if active else self.STOP if stop_latched else self.OUTLINE
        draw.polygon(body,fill=self.PANEL)
        draw.line(body+[body[0]],fill=body_outline,width=3,joint="curve")
        draw.line(((60,67),(360,67)),fill=(50,150,205),width=2)

        draw.rounded_rectangle((48,38,104,57),radius=7,fill=self.ACTIVE if lt else self.MUTED,outline=self.OUTLINE)
        self._centered_text(draw,(76,47),"LT",fill=self.BACKGROUND if lt else self.TEXT)
        draw.rounded_rectangle((286,38,340,57),radius=7,fill=self.STOP if rb else self.MUTED,outline=self.OUTLINE)
        self._centered_text(draw,(313,47),"RB",fill=self.BACKGROUND if rb else self.TEXT)
        draw.rounded_rectangle((346,38,402,57),radius=7,fill=self.STOP if rt else self.MUTED,outline=self.OUTLINE)
        self._centered_text(draw,(374,47),"RT",fill=self.BACKGROUND if rt else self.TEXT)

        left_active=max(abs(lx),abs(ly))>self.deadzone
        right_active=max(abs(rx),abs(ry))>self.deadzone
        for center,stick_x,stick_y,is_active,label in (
            ((112,96),lx,ly,left_active,"L"),
            ((274,158),rx,ry,right_active,"R"),
        ):
            self._circle(draw,center,29,fill=(7,48,82),outline=(67,211,255),width=3)
            nub=(int(center[0]+stick_x*14),int(center[1]+stick_y*14))
            self._circle(draw,nub,14,fill=self.MOTION if is_active else (31,43,57),outline=self.TEXT,width=1)
            self._centered_text(draw,center,label,fill=self.TEXT)

        draw.rounded_rectangle((78,142,140,166),radius=4,fill=(21,38,56),outline=self.OUTLINE)
        draw.rounded_rectangle((97,123,121,185),radius=4,fill=(21,38,56),outline=self.OUTLINE)
        if dy < -0.5:draw.rounded_rectangle((98,124,120,150),radius=3,fill=self.MOTION)
        if dy > 0.5:draw.rounded_rectangle((98,158,120,184),radius=3,fill=self.MOTION)
        if dx < -0.5:draw.rounded_rectangle((79,143,107,165),radius=3,fill=self.MOTION)
        if dx > 0.5:draw.rounded_rectangle((111,143,139,165),radius=3,fill=self.MOTION)
        self._circle(draw,(109,154),7,fill=self.MUTED)

        face=((342,84,"Y",self.YELLOW if y else (55,66,82),y),
              (342,142,"A",self.ACTIVE if a else (55,66,82),a),
              (313,113,"X",(55,66,82),False),(371,113,"B",(55,66,82),False))
        for px,py,label,color,pressed in face:
            self._circle(draw,(px,py),15,fill=color,outline=self.TEXT if pressed else self.OUTLINE,width=2)
            self._centered_text(draw,(px,py),label,fill=self.BACKGROUND if pressed else self.TEXT)

        self._circle(draw,(210,88),19,fill=(15,55,88),outline=(71,170,220),width=2)
        draw.polygon(((201,82),(210,94),(219,82),(210,87)),outline=self.TEXT)
        for px,label in ((181,"□"),(239,"≡")):
            self._circle(draw,(px,117),10,fill=(20,57,87),outline=(57,145,194))
            self._centered_text(draw,(px,117),label,fill=self.TEXT,bold=False)
        return np.asarray(image,dtype=np.uint8).copy()
