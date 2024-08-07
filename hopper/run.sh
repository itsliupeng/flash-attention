python setup.py clean
python setup.py bdist_wheel
pip uninstall -y flashattn-hopper && pip install dist/flashattn_hopper-3.0.0b1-cp310-cp310-linux_x86_64.whl
