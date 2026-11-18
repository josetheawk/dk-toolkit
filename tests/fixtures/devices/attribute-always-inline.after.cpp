struct Car {
    int m0;
    int m4;
    float m8;
    Car* next;
    int arr[8];
};
extern int g_count;
extern int Work(int);
extern void Poke(Car*, int);
inline int ClampPct(int v) {
    if (v < 0) v = 0;
    if (v > 100) v = 100;
    return v;
}
inline void SetPair(Car* c, int a, int b) {
    c->m0 = a;
    c->m4 = b;
}
inline __attribute__((always_inline)) int Work2(int v) {
    return Work(v) + 1;
}
int Func(Car* c) {
    return Work2(c->m0);
}
