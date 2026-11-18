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
void Func(Car* c, int a) {
    Car* d = c->next;
    SetPair(d, a, 7);
    Poke(c, a);
}
