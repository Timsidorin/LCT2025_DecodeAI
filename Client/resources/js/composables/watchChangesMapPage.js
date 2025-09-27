import {watch} from "vue";

export function useWatchRegion(storeRegion, func) {
    watch(
        () => storeRegion.region?.value,
        async (newValue, oldValue) => {
            if (newValue !== oldValue) {
                await func();
            }
        }
    );
}

export function useWatchStartDate(storeDate, func) {
    watch(
        () => storeDate.startDate,
        async (newId, oldId) => {
            if (newId !== oldId) {
                await func();
            }
        }
    );
}

export function useWatchEndDate(storeDate, func) {
    watch(
        () => storeDate.endDate,
        async (newId, oldId) => {
            if (newId !== oldId) {
                await func();
            }
        }
    );
}
