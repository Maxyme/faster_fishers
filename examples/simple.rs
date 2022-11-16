use faster_fishers::{fishers_exact_with_odds_ratio, Alternative};

fn main() {
    let table = [3, 5, 4, 50];
    let alternative = Alternative::Less;
    let (p_value, odds_ratio) = fishers_exact_with_odds_ratio(&table, alternative).unwrap();
    println!("p value: {}, odds ratio: {}", p_value, odds_ratio);
}
