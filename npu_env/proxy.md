ssh -fNgR 7333:127.0.0.1:7890 root@192.168.9.225


```bash
#YJH proxy
export proxy_addr=localhost
export proxy_http_port=7890
export proxy_socks_port=7890
function set_proxy() {
      export http_proxy=http://$proxy_addr:$proxy_http_port #如果使用git 不行，这两个http和https改成socks5就行
      export https_proxy=http://$proxy_addr:$proxy_http_port
      git config --global https.proxy http://$proxy_addr:$proxy_http_port
      git config --global https.proxy https://$proxy_addr:$proxy_http_port
      export all_proxy=socks5://$proxy_addr:$proxy_socks_port
      export no_proxy=127.0.0.1,.huawei.com,localhost,local,.local
}
function unset_proxy() {
      git config --global --unset http.proxy
      git config --global --unset https.proxy
      unset http_proxy
      unset https_proxy
      unset all_proxy
}
function test_proxy() {
      curl -v -x http://$proxy_addr:$proxy_http_port https://www.google.com | egrep 'HTTP/(2|1.1) 200'
      # socks5h://$proxy_addr:$proxy_socks_port
}
# set_proxy # 如果要登陆时默认启用代理则取消注释这句
```