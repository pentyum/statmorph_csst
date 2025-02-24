# statmorph_csst

### 1. 介绍
statmorph CSST专用版，采用cython进行优化

地址：[Gitee](https://gitee.com/pentyum/statmorph_csst) ｜ [Github](https://github.com/pentyum/statmorph_csst)

### 2. 安装教程

#### 2.1. 安装

克隆本仓库，进入statmorph_csst文件夹。

安装依赖，依赖包括：`numpy`, `scipy`, `astropy`, `photutils`, `scikit-image`, `Cython`, `setuptools`。

也可以用 `pip install -r requirements.txt` 来安装。

#### 2.2. 构建

运行`./build.sh`

### 3. 使用说明

1.  进入src目录，然后运行main.py
2.  `python3 main.py -h`进入帮助

```
SExtractor-Statmorph_csst 简化合并版使用说明
        -C, --config=配置文件名，默认为config.properties，配置同样可以通过命令行设置，通过命令行设置的的参数会覆盖配置文件中的参数
        -j, --threads=并行进程数量，若为0则为CPU核心数量-1(若为单核则为1)
        -i, --image_file=原始图像文件(未扣除背景)，双图像模式中指用来探测源的图像文件(深度越深，PSF越大越好)，若跳过SExtractor可以不需要
        -y, --measure_file=双图像模式中用于测量的图像文件(未扣除背景)，若不指定(为null)则为单图像模式，若跳过SExtractor 可以不需要
        -w, --wht_file=权重图像文件，若跳过SExtractor可以不需要
        -o, --save_file=形态学参数输出文件名，若不指定则默认为image_file的文件名(不包括后缀).txt，双图像模式则还包括measure_file
        -p, --run_percentage=运行全部源数量的百分比，100表示全部运行
        -l, --run_specified_label=仅运行指定编号的源，若为0则运行全部源
        -s, --sextractor_work_dir=SExtractor的输出文件存放文件夹，若不指定则默认为image_file的文件名(不包括后缀)，双图像模式下默认文件名还会包括measure_file，如果跳过sextractor，那么必须指定该项
        -k, --skip_sextractor 是否直接利用SExtractor已经生成的结果，SExtractor的输出文件夹必须包含subback.fits,segmap.fits,noise.fits三个图像文件和catalog.txt星表文件，星表文件至少包含三列，分别为NUMBER, MAG_AUTO和CLASS_STAR。NUMBER指的就是segmap中的label。
        -X, --sextractor_command=运行SExtractor的命令，默认为source-extractor，如果采用的是不同的方式安装，可能需要配置此设置。
        -D, --sextractor_detect_minarea
        -T, --sextractor_detect_thresh
        -A, --sextractor_analysis_thresh
        -B, --sextractor_deblend_nthresh
        -M, --sextractor_deblend_mincont
        -S, --sextractor_back_size
        -F, --sextractor_back_filtersize
        -P, --sextractor_backphoto_thick
        -r, --stamp_catalog 如果填写则进入stamp模式，每个星系具有独立的stamp的fits文件，而不是从segmap中创建，stamp_catalog文件必须包含label，image_file_name，image_hdu_index，(noise_file_name，noise_hdu_index，cmp_file_name，cmp_hdu_index)列 ，如果不指定hdu_index，则默认为0。指定该项后，image_file、measure_file、wht_file、sextractor_work_dir、skip_sextractor将全部失效。
        -a, --output_image_dir=输出示意图的文件夹，若为null则不输出示意图
        -f, --ignore_mag_fainter_than=忽略测量视星等比该星等更高的源
        -t, --ignore_class_star_greater_than=忽略测量像恒星指数大于该值的源
        -n, --center_file=预定义的星系中心文件，用于取代星系质心和不对称中心。需要列centroid_x,centroid_y,asymmetry_center_x,asymmetry_center_y
        -c, --calc_cas 是否测量CAS
        -g, --calc_g_m20 是否测量Gini,M20
        -x, --calc_shape_asymmetry 是否测量shape_asymmetry，依赖CAS和G_M20
        -d, --calc_mid 是否测量MID
        -u, --calc_multiplicity 是否测量multiplicity
        -e, --calc_color_dispersion 是否测量color_dispersion
        -m, --image_compare_file=测量color_dispersion中用于比较的图像(已经扣除了背景)，若不测量则为null
        -b, --calc_g2 是否测量G2
        -v, --use_vanilla 是否使用vanilla版
        -h, --help 显示此帮助
```

#### 3.1. 配置文件
默认配置文件为src/config.properties，内容和命令行参数的配置内容一样，命令行的配置将覆盖配置文件中的配置。

通过-C可以指定配置文件路径。

### 4. 程序更新
进入src目录，运行`./update.sh`，建议先运行`./clean.sh`清理掉`.c`和`.so`文件后再更新。
