#!/bin/bash

# 检查是否传入至少一个文件或选项
if [ $# -lt 1 ]; then
    echo "Usage: $0 [--hour <h>] [--minute <m>] [--second <s>] <python_script1> [args_for_script1...] [-- <python_script2> [args_for_script2...] ...]"
    exit 1
fi

# 初始化倒计时参数
HOUR=0
MINUTE=0
SECOND=0

# 初始化一个标志，指示是否已经开始解析脚本部分
PARSE_SCRIPTS=false

# 初始化数组来存储脚本和它们的参数
scripts=()
args=()
current_script=""
current_args=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    if [[ "$PARSE_SCRIPTS" == false && "$1" == "--hour" ]]; then
        shift
        if [[ "$1" =~ ^[0-9]+$ ]]; then
            HOUR=$1
        else
            echo "Error: --hour 参数需要一个整数值."
            exit 1
        fi
        shift
    elif [[ "$PARSE_SCRIPTS" == false && "$1" == "--minute" ]]; then
        shift
        if [[ "$1" =~ ^[0-9]+$ ]]; then
            MINUTE=$1
        else
            echo "Error: --minute 参数需要一个整数值."
            exit 1
        fi
        shift
    elif [[ "$PARSE_SCRIPTS" == false && "$1" == "--second" ]]; then
        shift
        if [[ "$1" =~ ^[0-9]+$ ]]; then
            SECOND=$1
        else
            echo "Error: --second 参数需要一个整数值."
            exit 1
        fi
        shift
    elif [[ "$1" == "--" ]]; then
        # 遇到 '--' 后，开始解析脚本部分
        PARSE_SCRIPTS=true
        shift
    else
        # 开始解析脚本部分
        PARSE_SCRIPTS=true
        if [[ -z "$current_script" ]]; then
            current_script=$1
        else
            current_args+="$1 "
        fi
        shift
    fi

    # 如果已经开始解析脚本部分且遇到 '--'，将当前脚本和参数加入数组
    if [[ "$PARSE_SCRIPTS" == true && "$1" == "--" ]]; then
        if [[ -n "$current_script" ]]; then
            scripts+=("$current_script")
            args+=("$current_args")
            current_script=""
            current_args=""
        fi
        shift  # 移除 '--'
    fi
done

# 处理最后一组脚本及参数
if [[ -n "$current_script" ]]; then
    scripts+=("$current_script")
    args+=("$current_args")
fi

# 检查是否有至少一个脚本要执行
if [ ${#scripts[@]} -lt 1 ]; then
    echo "Error: 未指定要执行的 Python 脚本."
    exit 1
fi

# 计算总秒数
TOTAL_SECONDS=$((HOUR * 3600 + MINUTE * 60 + SECOND))

# 执行倒计时
if [[ $TOTAL_SECONDS -gt 0 ]]; then
    echo "倒计时开始: $HOUR 小时 $MINUTE 分 $SECOND 秒"
    while [ $TOTAL_SECONDS -gt 0 ]; do
        hrs=$((TOTAL_SECONDS / 3600))
        mins=$(( (TOTAL_SECONDS % 3600) / 60 ))
        secs=$((TOTAL_SECONDS % 60 ))
        printf "\r倒计时: %02d:%02d:%02d" $hrs $mins $secs
        sleep 1
        TOTAL_SECONDS=$((TOTAL_SECONDS - 1))
    done
    echo -e "\n倒计时结束，开始执行脚本..."
fi

# 遍历每个 Python 脚本及其对应参数，依次运行
for i in "${!scripts[@]}"; do
    script=${scripts[$i]}
    script_args=${args[$i]}

    # 检查 Python 文件是否存在
    if [ ! -f "$script" ]; then
        echo "Error: Python script '$script' not found."
        exit 1
    fi

    # 无限循环运行当前脚本，直到成功退出
    while true; do
        echo "运行的 Python 文件是: $script"
        echo "传入的参数是: $script_args"

        # 执行 Python 脚本
        python "$script" $script_args

        # 检查脚本的退出码
        if [ $? -eq 0 ]; then
            echo "$script 运行完成, 正常退出循环"
            break  # 正常退出循环
        else
            echo "$script 运行错误, 等待三秒重启..."
            printf '%.0s-' {1..150}
            echo
            sleep 3
        fi
    done
done