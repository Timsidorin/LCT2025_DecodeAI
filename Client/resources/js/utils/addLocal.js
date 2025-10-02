//и такие костыли бывают
export function addLocal() {
    let city = localStorage.getItem('selectedCity');
    let region = localStorage.getItem('selectedRegion');
    if (!city) {
        localStorage.setItem('selectedCity', '{"label":"Москва","value":"Москва"}')
    }
    if (!region) {
        localStorage.setItem('selectedRegion', '{"label":"г Москва","value":"RU-MOW","total_reviews":12,"cities":[{"city_name":"Москва","reviews_count":12}]}')
    }
}
