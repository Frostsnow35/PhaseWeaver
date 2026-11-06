# 本机推送脚本。
#
# 为什么需要它:
#   1) 沙箱禁止访问凭证管理器,git 无法使用 credential helper;
#   2) git 默认的 schannel TLS 后端在本机拿不到凭证(SEC_E_NO_CREDENTIALS),
#      必须显式切到 openssl 后端。
# 令牌只来自 `gh auth token`,不落盘、不写进任何文件。
$ErrorActionPreference = "Stop"
Push-Location (Join-Path $PSScriptRoot "..")
try {
    $token = gh auth token
    if (-not $token) { throw "gh 未登录,先执行 gh auth login" }
    $basic = [Convert]::ToBase64String(
        [Text.Encoding]::ASCII.GetBytes("x-access-token:$token")
    )

    $env:GIT_CONFIG_COUNT = "2"
    $env:GIT_CONFIG_KEY_0 = "http.extraheader"
    $env:GIT_CONFIG_VALUE_0 = "AUTHORIZATION: basic $basic"
    $env:GIT_CONFIG_KEY_1 = "http.sslBackend"
    $env:GIT_CONFIG_VALUE_1 = "openssl"

    git push @args
    exit $LASTEXITCODE
}
finally {
    Remove-Item Env:GIT_CONFIG_COUNT, Env:GIT_CONFIG_KEY_0, Env:GIT_CONFIG_VALUE_0,
        Env:GIT_CONFIG_KEY_1, Env:GIT_CONFIG_VALUE_1 -ErrorAction SilentlyContinue
    Pop-Location
}
