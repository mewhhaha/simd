use std::env;

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> simd::Result<()> {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        return Err(simd::SimdError::new(
            "usage: simd <parse|check|fmt|run|run-profile|wasm|wat|run-wasm|run-wasm-prepared|run-wasm-profile|bench|inspect-html|lsp> ...",
        ));
    }
    let command = args.remove(0);
    match command.as_str() {
        "parse" => {
            let path = args
                .first()
                .ok_or_else(|| simd::SimdError::new("usage: simd parse <file>"))?;
            println!("{}", simd::parse_command(path)?);
        }
        "check" => {
            let path = args
                .first()
                .ok_or_else(|| simd::SimdError::new("usage: simd check <file>"))?;
            println!("{}", simd::check_command(path)?);
        }
        "fmt" => {
            if args.is_empty() {
                return Err(simd::SimdError::new("usage: simd fmt <file> [--check]"));
            }
            let path = args.remove(0);
            let mut check = false;
            let mut index = 0usize;
            while index < args.len() {
                match args[index].as_str() {
                    "--check" => {
                        check = true;
                        index += 1;
                    }
                    flag => {
                        return Err(simd::SimdError::new(format!("unknown fmt flag '{}'", flag)));
                    }
                }
            }
            println!("{}", simd::fmt_command(&path, check)?);
        }
        "run" => {
            if args.is_empty() {
                return Err(simd::SimdError::new(
                    "usage: simd run <file> --main <fn> --args <json>",
                ));
            }
            let path = args.remove(0);
            let mut main = None::<String>;
            let mut json = None::<String>;
            let mut index = 0usize;
            while index < args.len() {
                match args[index].as_str() {
                    "--main" => {
                        let value = args
                            .get(index + 1)
                            .ok_or_else(|| simd::SimdError::new("missing value after --main"))?;
                        main = Some(value.clone());
                        index += 2;
                    }
                    "--args" => {
                        let value = args
                            .get(index + 1)
                            .ok_or_else(|| simd::SimdError::new("missing value after --args"))?;
                        json = Some(value.clone());
                        index += 2;
                    }
                    flag => {
                        return Err(simd::SimdError::new(format!("unknown run flag '{}'", flag)));
                    }
                }
            }
            let main = main.ok_or_else(|| simd::SimdError::new("run requires --main <fn>"))?;
            let json = json.ok_or_else(|| simd::SimdError::new("run requires --args <json>"))?;
            println!("{}", simd::run_command(&path, &main, &json)?);
        }
        "run-profile" => {
            if args.is_empty() {
                return Err(simd::SimdError::new(
                    "usage: simd run-profile <file> --main <fn> --args <json>",
                ));
            }
            let path = args.remove(0);
            let mut main = None::<String>;
            let mut json = None::<String>;
            let mut index = 0usize;
            while index < args.len() {
                match args[index].as_str() {
                    "--main" => {
                        let value = args
                            .get(index + 1)
                            .ok_or_else(|| simd::SimdError::new("missing value after --main"))?;
                        main = Some(value.clone());
                        index += 2;
                    }
                    "--args" => {
                        let value = args
                            .get(index + 1)
                            .ok_or_else(|| simd::SimdError::new("missing value after --args"))?;
                        json = Some(value.clone());
                        index += 2;
                    }
                    flag => {
                        return Err(simd::SimdError::new(format!(
                            "unknown run-profile flag '{}'",
                            flag
                        )));
                    }
                }
            }
            let main = main.ok_or_else(|| simd::SimdError::new("run-profile requires --main <fn>"))?;
            let json = json.ok_or_else(|| simd::SimdError::new("run-profile requires --args <json>"))?;
            println!("{}", simd::run_profile_command(&path, &main, &json)?);
        }
        "wasm" => {
            if args.is_empty() {
                return Err(simd::SimdError::new(
                    "usage: simd wasm <file> --main <fn> [--out <path>]",
                ));
            }
            let path = args.remove(0);
            let mut main = None::<String>;
            let mut out = None::<String>;
            let mut index = 0usize;
            while index < args.len() {
                match args[index].as_str() {
                    "--main" => {
                        let value = args
                            .get(index + 1)
                            .ok_or_else(|| simd::SimdError::new("missing value after --main"))?;
                        main = Some(value.clone());
                        index += 2;
                    }
                    "--out" => {
                        let value = args
                            .get(index + 1)
                            .ok_or_else(|| simd::SimdError::new("missing value after --out"))?;
                        out = Some(value.clone());
                        index += 2;
                    }
                    flag => {
                        return Err(simd::SimdError::new(format!(
                            "unknown wasm flag '{}'",
                            flag
                        )));
                    }
                }
            }
            let main = main.ok_or_else(|| simd::SimdError::new("wasm requires --main <fn>"))?;
            println!("{}", simd::wasm_command(&path, &main, out.as_deref())?);
        }
        "wat" => {
            if args.is_empty() {
                return Err(simd::SimdError::new("usage: simd wat <file> --main <fn>"));
            }
            let path = args.remove(0);
            let mut main = None::<String>;
            let mut index = 0usize;
            while index < args.len() {
                match args[index].as_str() {
                    "--main" => {
                        let value = args
                            .get(index + 1)
                            .ok_or_else(|| simd::SimdError::new("missing value after --main"))?;
                        main = Some(value.clone());
                        index += 2;
                    }
                    flag => {
                        return Err(simd::SimdError::new(format!("unknown wat flag '{}'", flag)));
                    }
                }
            }
            let main = main.ok_or_else(|| simd::SimdError::new("wat requires --main <fn>"))?;
            println!("{}", simd::wat_command(&path, &main)?);
        }
        "run-wasm" => {
            if args.is_empty() {
                return Err(simd::SimdError::new(
                    "usage: simd run-wasm <file> --main <fn> --args <json>",
                ));
            }
            let path = args.remove(0);
            let mut main = None::<String>;
            let mut json = None::<String>;
            let mut index = 0usize;
            while index < args.len() {
                match args[index].as_str() {
                    "--main" => {
                        let value = args
                            .get(index + 1)
                            .ok_or_else(|| simd::SimdError::new("missing value after --main"))?;
                        main = Some(value.clone());
                        index += 2;
                    }
                    "--args" => {
                        let value = args
                            .get(index + 1)
                            .ok_or_else(|| simd::SimdError::new("missing value after --args"))?;
                        json = Some(value.clone());
                        index += 2;
                    }
                    flag => {
                        return Err(simd::SimdError::new(format!(
                            "unknown run-wasm flag '{}'",
                            flag
                        )));
                    }
                }
            }
            let main = main.ok_or_else(|| simd::SimdError::new("run-wasm requires --main <fn>"))?;
            let json =
                json.ok_or_else(|| simd::SimdError::new("run-wasm requires --args <json>"))?;
            println!("{}", simd::run_wasm_command(&path, &main, &json)?);
        }
        "run-wasm-prepared" => {
            if args.is_empty() {
                return Err(simd::SimdError::new(
                    "usage: simd run-wasm-prepared <file> --main <fn> --args <json> [--iters <n>]",
                ));
            }
            let path = args.remove(0);
            let mut main = None::<String>;
            let mut json = None::<String>;
            let mut iters = 1usize;
            let mut index = 0usize;
            while index < args.len() {
                match args[index].as_str() {
                    "--main" => {
                        let value = args
                            .get(index + 1)
                            .ok_or_else(|| simd::SimdError::new("missing value after --main"))?;
                        main = Some(value.clone());
                        index += 2;
                    }
                    "--args" => {
                        let value = args
                            .get(index + 1)
                            .ok_or_else(|| simd::SimdError::new("missing value after --args"))?;
                        json = Some(value.clone());
                        index += 2;
                    }
                    "--iters" => {
                        let value = args
                            .get(index + 1)
                            .ok_or_else(|| simd::SimdError::new("missing value after --iters"))?;
                        iters = value.parse::<usize>().map_err(|error| {
                            simd::SimdError::new(format!(
                                "invalid --iters value '{}': {}",
                                value, error
                            ))
                        })?;
                        index += 2;
                    }
                    flag => {
                        return Err(simd::SimdError::new(format!(
                            "unknown run-wasm-prepared flag '{}'",
                            flag
                        )));
                    }
                }
            }
            let main = main
                .ok_or_else(|| simd::SimdError::new("run-wasm-prepared requires --main <fn>"))?;
            let json = json.ok_or_else(|| {
                simd::SimdError::new("run-wasm-prepared requires --args <json>")
            })?;
            println!(
                "{}",
                simd::run_wasm_prepared_command(&path, &main, &json, iters)?
            );
        }
        "run-wasm-profile" => {
            if args.is_empty() {
                return Err(simd::SimdError::new(
                    "usage: simd run-wasm-profile <file> --main <fn> --args <json>",
                ));
            }
            let path = args.remove(0);
            let mut main = None::<String>;
            let mut json = None::<String>;
            let mut index = 0usize;
            while index < args.len() {
                match args[index].as_str() {
                    "--main" => {
                        let value = args
                            .get(index + 1)
                            .ok_or_else(|| simd::SimdError::new("missing value after --main"))?;
                        main = Some(value.clone());
                        index += 2;
                    }
                    "--args" => {
                        let value = args
                            .get(index + 1)
                            .ok_or_else(|| simd::SimdError::new("missing value after --args"))?;
                        json = Some(value.clone());
                        index += 2;
                    }
                    flag => {
                        return Err(simd::SimdError::new(format!(
                            "unknown run-wasm-profile flag '{}'",
                            flag
                        )));
                    }
                }
            }
            let main =
                main.ok_or_else(|| simd::SimdError::new("run-wasm-profile requires --main <fn>"))?;
            let json = json.ok_or_else(|| {
                simd::SimdError::new("run-wasm-profile requires --args <json>")
            })?;
            println!("{}", simd::run_wasm_profile_command(&path, &main, &json)?);
        }
        "bench" => {
            let (mut selection, mut index) = match args.first() {
                Some(first) if !first.starts_with("--") => (first.clone(), 1usize),
                _ => ("all".to_string(), 0usize),
            };
            let mut size = 0usize;
            let mut iterations = None::<usize>;
            let mut report_contract = true;
            while index < args.len() {
                match args[index].as_str() {
                    "--size" => {
                        let value = args
                            .get(index + 1)
                            .ok_or_else(|| simd::SimdError::new("missing value after --size"))?;
                        size = value.parse::<usize>().map_err(|error| {
                            simd::SimdError::new(format!(
                                "invalid --size value '{}': {}",
                                value, error
                            ))
                        })?;
                        index += 2;
                    }
                    "--iters" => {
                        let value = args
                            .get(index + 1)
                            .ok_or_else(|| simd::SimdError::new("missing value after --iters"))?;
                        iterations = Some(value.parse::<usize>().map_err(|error| {
                            simd::SimdError::new(format!(
                                "invalid --iters value '{}': {}",
                                value, error
                            ))
                        })?);
                        index += 2;
                    }
                    "--matrix" => {
                        selection = "matrix".to_string();
                        index += 1;
                    }
                    "--no-contract" => {
                        report_contract = false;
                        index += 1;
                    }
                    flag => {
                        return Err(simd::SimdError::new(format!(
                            "unknown bench flag '{}'",
                            flag
                        )));
                    }
                }
            }
            println!(
                "{}",
                simd::bench_command_with_options(simd::BenchOptions {
                    selection,
                    size,
                    iterations,
                    report_contract,
                })?
            );
        }
        "inspect-html" => {
            let mut out = None::<String>;
            let mut index = 0usize;
            while index < args.len() {
                match args[index].as_str() {
                    "--out" => {
                        let value = args
                            .get(index + 1)
                            .ok_or_else(|| simd::SimdError::new("missing value after --out"))?;
                        out = Some(value.clone());
                        index += 2;
                    }
                    flag => {
                        return Err(simd::SimdError::new(format!(
                            "unknown inspect-html flag '{}'",
                            flag
                        )));
                    }
                }
            }
            println!("{}", simd::inspect_html_command(out.as_deref())?);
        }
        "lsp" => {
            if !args.is_empty() {
                return Err(simd::SimdError::new("usage: simd lsp"));
            }
            simd::lsp_command()?;
        }
        other => {
            return Err(simd::SimdError::new(format!("unknown command '{}'", other)));
        }
    }
    Ok(())
}
