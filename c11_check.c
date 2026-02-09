#include "duckdb_flow.h"

/* Verify key types and functions are accessible from C11 */
void c11_smoke(void) {
    ColType t = COL_INT32;
    (void)t;
    (void)sizeof(DoubleBuf);
    (void)sizeof(Slot);
    (void)sizeof(Batch);
    (void)sizeof(Schema);
    (void)sizeof(ColDef);
}
