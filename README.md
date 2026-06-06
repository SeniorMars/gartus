# gartus

I finished this some time ago. But i just added a bunch of features.

- Work with images made outside this engine and manuipulate them
- Filters
- HSL support
- turtle support	
- And of course: curves :)

## Ray tracing examples

Use the render profile for full path-traced examples:

```bash
cargo run --profile render --example life
```

For iteration, lower the image width and samples/grid width in the example first. Random sampling
can also use `RayCamera::with_adaptive_sampling(min, max, threshold)` to stop converged pixels
before the maximum sample count.

![Corro](./corro.png)
