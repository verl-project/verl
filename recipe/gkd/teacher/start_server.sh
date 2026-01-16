export PROXY_FRONTEND_PORT=15555
export PROXY_BACKEND_PORT=15556

export CUDA_VISIBLE_DEVICES=2
export CUDA_DEVICE_ORDER=PCI_BUS_ID


BACKEND=vllm
# CKPT_PATH="/path/to/TEACHER_MODEL/"
CKPT_PATH="Qwen/Qwen2.5-1.5B-Instruct" 

wait_server_ready() {
    server=$1
    ip=$2
    port=$3
    while true; do
        echo "wait $server server ready at $ip:$port..."
        result=`echo -e "\n" | telnet $ip $port 2> /dev/null | grep Connected | wc -l`
        if [ $result -eq 1 ]; then
            break
        else
            sleep 1
        fi
    done
}

ps -ef | grep "python proxy.py" | grep -v grep | awk -F ' ' '{print $2}' | xargs -r kill -9
ps -ef | grep "python worker.py" | grep -v grep | awk -F ' ' '{print $2}' | xargs -r kill -9

nohup python proxy.py &> proxy.log &

wait_server_ready proxy localhost $PROXY_BACKEND_PORT

echo "teacher proxy is ready"

nohup python worker.py --backend $BACKEND --tp-size 1 --n-logprobs 256 --ckpt-path $CKPT_PATH &> worker.log &
echo "start teacher worker"

echo "teacher server is ready"