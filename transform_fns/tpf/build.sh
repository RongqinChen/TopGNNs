cd transform_fns/tpf
python setup.py build_ext --inplace

rm -rf build

cd ../../
python -m transform_fns.to_tpf
