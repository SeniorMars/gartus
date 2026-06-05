//! Handwritten MDL lexer.

use super::diagnostic::Diagnostic;

/// A token source span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    /// One-based source line.
    pub line: usize,
    /// One-based starting source column.
    pub col_start: usize,
    /// One-based ending source column.
    pub col_end: usize,
}

/// Kind of MDL token.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    /// A non-numeric word token.
    Word(String),
    /// A finite numeric token.
    Number(f64),
    /// A mesh filename token introduced by `:`.
    Filename(String),
}

/// One lexed MDL token.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    /// Token payload.
    pub kind: TokenKind,
    /// Source span.
    pub span: Span,
}

/// Lexes one MDL source line.
///
/// # Errors
/// Returns a diagnostic when a filename token is missing its path or a numeric token is not finite.
pub fn lex_line(line_no: usize, line: &str) -> Result<Vec<Token>, Diagnostic> {
    let mut tokens = Vec::new();
    let mut index = 0;

    while index < line.len() {
        let Some(ch) = line[index..].chars().next() else {
            break;
        };
        if ch.is_whitespace() {
            index += ch.len_utf8();
            continue;
        }

        let start = index;
        while index < line.len() {
            let Some(ch) = line[index..].chars().next() else {
                break;
            };
            if ch.is_whitespace() {
                break;
            }
            index += ch.len_utf8();
        }

        let raw = &line[start..index];
        if raw.starts_with("//") {
            break;
        }

        let span = Span {
            line: line_no,
            col_start: char_col(line, start),
            col_end: char_col_end(line, index),
        };

        let kind = if let Some(path) = raw.strip_prefix(':') {
            if path.is_empty() {
                return Err(Diagnostic::new(
                    line_no,
                    span.col_start,
                    span.col_end,
                    "expected filename after `:`",
                ));
            }
            TokenKind::Filename(path.to_string())
        } else if let Ok(value) = raw.parse::<f64>() {
            if !value.is_finite() {
                return Err(Diagnostic::new(
                    line_no,
                    span.col_start,
                    span.col_end,
                    "number must be finite",
                ));
            }
            TokenKind::Number(value)
        } else {
            TokenKind::Word(raw.to_string())
        };

        tokens.push(Token { kind, span });
    }

    Ok(tokens)
}

fn char_col(line: &str, byte_index: usize) -> usize {
    line[..byte_index].chars().count() + 1
}

fn char_col_end(line: &str, byte_index: usize) -> usize {
    line[..byte_index].chars().count()
}

#[cfg(test)]
mod tests {
    use super::{TokenKind, lex_line};

    #[test]
    fn lexes_numbers_words_and_mesh_filenames() {
        let tokens = lex_line(3, "mesh metal :teapot.obj world // comment").unwrap();

        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].kind, TokenKind::Word("mesh".to_string()));
        assert_eq!(tokens[1].kind, TokenKind::Word("metal".to_string()));
        assert_eq!(
            tokens[2].kind,
            TokenKind::Filename("teapot.obj".to_string())
        );
        assert_eq!(tokens[3].span.line, 3);
        assert_eq!(tokens[3].span.col_start, 24);
    }

    #[test]
    fn rejects_empty_filename_token() {
        let error = lex_line(1, "mesh :").unwrap_err();

        assert_eq!(error.line, 1);
        assert!(error.message.contains("filename"));
    }

    #[test]
    fn comment_marker_only_starts_comment_as_token() {
        let tokens = lex_line(1, "save out//foo.png // comment").unwrap();

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[1].kind, TokenKind::Word("out//foo.png".to_string()));
    }

    #[test]
    fn spans_count_unicode_chars_not_bytes() {
        let tokens = lex_line(1, "é move").unwrap();

        assert_eq!(tokens[1].span.col_start, 3);
        assert_eq!(tokens[1].span.col_end, 6);
    }
}
