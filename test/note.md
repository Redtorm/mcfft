# matrix core 乘法规则
若计算C = A * B，其中A B C均为16*16的矩阵，A B C声明如下：
```c++
rocwmma::fragment<rocwmma::accumulator, rocwmma_M, rocwmma_N, rocwmma_K, Myhalf> matrixC;
rocwmma::fragment<rocwmma::matrix_a, rocwmma_M, rocwmma_N, rocwmma_K, Myhalf, rocwmma::row_major> matrixA;
rocwmma::fragment<rocwmma::matrix_b, rocwmma_M, rocwmma_N, rocwmma_K, Myhalf, rocwmma::col_major> matrixB;
```
按如下方式从内存载入数据，在MI100GPU中内存数据每64个线程为一个warp，每个warp计算两个个16*16矩阵的乘加。每个线程掌控256/64=4个数据。
```c++
rocwmma::load_matrix_sync(matrixA, inA, 16);
rocwmma::load_matrix_sync(matrixB, inB, 16);
```
若inA和inB中的数据为1~256，则threadIdx.x=0的线程掌握matrixA的4个数据为1 2 3 4，matrixB的4个数据为1 2 3 4。掌握的计算结果matrixC的4个数据为1496 3672 5848 8024。这个结果为对应行相乘，并不符合矩阵乘法定义，所以要想实现矩阵相乘应该将矩阵B转置然后读入matrixB。