#include <bits/stdc++.h>
using namespace std;
using int128 = __int128_t;

const long long MOD = 1000000007;
const long long INV2 = 500000004; // (1/2) mod MOD，因為 2 * 500000004 ≡ 1 (mod 1e9+7)

// 幫助印出 __int128
void print128(int128 x)
{
    if (x == 0)
    {
        cout << 0;
        return;
    }
    string s;
    bool neg = false;
    if (x < 0)
    {
        neg = true;
        x = -x;
    }
    while (x)
    {
        s.push_back('0' + x % 10);
        x /= 10;
    }
    if (neg)
        cout << '-';
    reverse(s.begin(), s.end());
    cout << s;
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n;
    cin >> n;

    int128 ans = 0;
    long long last = 0;

    // ---- 第一段：i <= sqrt(n) ----
    for (long long i = 1; i * i <= n; ++i)
    {
        long long term = (n / i) % MOD;
        ans = (ans + (int128)term * (i % MOD)) % MOD;
        last = n / i;
    }

    // ---- 第二段：枚舉商 v ----
    for (long long v = last - 1; v >= 1; --v)
    {
        long long L = n / (v + 1) + 1;
        long long R = n / v;

        long long cnt = (R - L + 1) % MOD;
        long long sumLR = ((L % MOD) + (R % MOD)) % MOD;
        long long seg = (int128)sumLR * cnt % MOD * INV2 % MOD; // (L+...+R) mod MOD
        ans = (ans + (int128)seg * (v % MOD)) % MOD;
    }

    long long result = (long long)(ans % MOD);
    cout << result << "\n";
    return 0;
}
