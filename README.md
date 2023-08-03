### build
```
git clone https://github.com/lturing/optical_flow.git
cd optical_flow
mkdir build
cd build
cmake .. && make -j6
cd ..

### test euroc
./build/opticalFlow ./config/euroc.yaml /home/spurs/dataset/euroc/MH_02_easy/mav0/cam0

```

### ref
- [rlp_vio](https://github.com/zju3dv/RLP_VIO)


