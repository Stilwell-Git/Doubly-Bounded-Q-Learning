#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <time.h>
#include <math.h>
#include <map>
using namespace std;

// Rabin-Karp rolling hash

long long Power(long long a,long long b,long long p)
{
    long long ans=1;
    for(;b;a=a*a%p,b>>=1)if(b&1)ans=ans*a%p;
    return ans;
}

namespace Primitive_root
{
    int pri[100],tot;
    bool check_prime(int g)
    {
        for(int i=2;i*i<=g;++i)if(g%i==0)return false;
        return true;
    }
    int calc(int p, int low)
    {
        // find a primitive root (>low) of given prime p
        srand(time(0));
        int i,j=p-1,g;tot=0;
        for(i=2;i*i<=j;++i)
            if(j%i==0)
                for(pri[++tot]=i;j%i==0;j/=i);
        if(j!=1)pri[++tot]=j;
        for(;;)
        {
            g = rand()%(p-1)+1;
            if(check_prime(g)&&(g>low))
            {
                for(i=1;i<=tot;++i)
                    if(Power(g,(p-1)/pri[i],p)==1)
                        break;
                if(i>tot) return g;
            }
        }
    }
}

#define W 84 // width
#define H 84 // height
#define C 4  // channels
#define P1 1000000007 // prime modulus of rolling hash
#define P2 1000000009 // prime modulus of rolling hash
const int b1 = Primitive_root::calc(P1, 100000); // base of rolling hash
const int b2 = Primitive_root::calc(P2, 100000); // base of rolling hash

#define L 233
map<long long, int> Hash_s[L];

#define D 10000005
#define gamma gamma_ // namespace conflict
double V_s[D], r_sas[D], gamma;
int head_s[D], next_sa[D], action_sa[D];
int head_sa[D], next_sas[D], sp_sas[D], count_sas[D];
int state_tot, state_action_tot, state_action_state_tot;

extern "C"
void init(double init_gamma)
{
    gamma = init_gamma;

    V_s[0] = 0.0;
    state_tot = 0;
    state_action_tot = 0;
    state_action_state_tot = 0;
    for (int i=0; i<L; ++i)
        Hash_s[i].clear();
}

long long calc_hash(unsigned char obs[W][H][C], int p, int b)
{
    long long h = 0;
    for (int i=0; i<W; ++i)
        for (int j=0; j<H; ++j)
            for (int k=0; k<C; ++k)
                h = (h*b+obs[i][j][k])%p;
    return h;
}

long long get_hash(unsigned char obs[W][H][C])
{
    long long h1 = calc_hash(obs, P1, b1);
    long long h2 = calc_hash(obs, P2, b2);
    long long h = h1*(P2+5)+h2;
    return h;
}

extern "C"
int get_state_tot()
{
    return state_tot;
}

extern "C"
int get_state_id(unsigned char obs[W][H][C])
{
    long long h = get_hash(obs);
    if (Hash_s[h%L].find(h)==Hash_s[h%L].end())
    {
        ++state_tot;
        V_s[state_tot] = 0.0;
        head_s[state_tot] = 0;
        Hash_s[h%L][h] = state_tot;
    }
    return Hash_s[h%L][h];
}

void add_transition_sa(int now_sa, double reward, int next_state_id)
{
    for (int now_sas = head_sa[now_sa]; now_sas; now_sas = next_sas[now_sas])
        if (sp_sas[now_sas]==next_state_id)
        {
            ++count_sas[now_sas];
            r_sas[now_sas] += reward;
            return;
        }
    int now_sas = ++state_action_state_tot;
    sp_sas[now_sas] = next_state_id;
    next_sas[now_sas] = head_sa[now_sa];
    head_sa[now_sa] = now_sas;
    count_sas[now_sas] = 1;
    r_sas[now_sas] = reward;
}

extern "C"
void add_transition(int state_id, int action, double reward, int next_state_id)
{
    for (int now_sa = head_s[state_id]; now_sa; now_sa = next_sa[now_sa])
        if (action_sa[now_sa]==action)
        {
            add_transition_sa(now_sa, reward, next_state_id);
            return;
        }

    int now_sa = ++state_action_tot;
    action_sa[now_sa] = action;
    next_sa[now_sa] = head_s[state_id];
    head_s[state_id] = now_sa;
    head_sa[now_sa] = 0;
    add_transition_sa(now_sa, reward, next_state_id);
}

double sqr(double x) { return x*x; }
double calc_Q_sa(int now_sa)
{
    int count_now = 0;
    for (int now_sas = head_sa[now_sa]; now_sas; now_sas = next_sas[now_sas])
        count_now += count_sas[now_sas];

    double Q_mean = 0.0, Q_min=1e9;
    for (int now_sas = head_sa[now_sa]; now_sas; now_sas = next_sas[now_sas])
    {
        double Q_now = r_sas[now_sas]/count_sas[now_sas]+gamma*V_s[sp_sas[now_sas]];
        Q_mean += Q_now*((double)count_sas[now_sas]/count_now);
        if (Q_now<Q_min) Q_min = Q_now;
    }

    return Q_mean;
}

extern "C"
void update_state(int state_id)
{
    // update the value of a single state
    double V_now = -(1e7);
    for (int now_sa = head_s[state_id]; now_sa; now_sa = next_sa[now_sa])
    {
        double Q_sa = calc_Q_sa(now_sa);
        if (Q_sa > V_now) V_now = Q_sa;
    }
    V_s[state_id] = V_now;
}

extern "C"
void update_buffer()
{
    // perform one iteration of value iteration
    for (int i=1; i<=state_tot; ++i)
        update_state(i);
}

extern "C"
double get_state_value(int state_id)
{
    return V_s[state_id];
}
