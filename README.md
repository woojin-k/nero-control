# nero-control
Python control code for the AgileX Nero robotic arm using the pyAgxArm SDK

## Run

```bash
# unified console (one prompt, one shared CAN connection)
./.venv/bin/python src/nero_ctrl/nero_cli.py --can can0 --fw v112

# GUI — same cores, same commands, popup confirms instead of console y/N
./.venv/bin/python src/nero_ctrl/nero_gui.py --can can0 --fw v112
```

The GUI is a thin shell over `nero_cli.dispatch()`: every button builds the
exact CLI command line, so waypoints and sequences are interchangeable between
the two frontends. Nothing moves without an explicit confirm popup.
