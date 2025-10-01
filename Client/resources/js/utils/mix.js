export function getFirstCharInUp(word) {
    let first = word[0].toUpperCase();
    let restOfString = word.slice(1);

    return first + restOfString;
}
