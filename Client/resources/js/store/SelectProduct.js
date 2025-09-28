import {defineStore} from "pinia";
import {ref} from "vue";

export const useProductStore = defineStore('product', () => {
    const product = ref(null);

    function setProduct(newProduct) {
        product.value = newProduct;
    }

    return {product, setProduct};
});
