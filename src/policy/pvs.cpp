#include <utility>
#include "state.hpp"
#include "pvs.hpp"

/*============================================================
 * PVS (Principal Variation Search) -- eval_ctx
 *
 * Negamax with PVS (NegaScout) and alpha-beta pruning.
 * Caller manages memory.
 * alpha: best score the maximizing side can already guarantee.
 * beta:  best score the minimizing side can already guarantee.
 *============================================================*/
 int PVS::eval_ctx(
    State *state,
    int depth,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const PVSParams& p,
    int alpha,
    int beta
){
    ctx.nodes++;
    if(ply > ctx.seldepth){
        ctx.seldepth = ply;
    }
    if(ctx.stop){
        return 0;
    }

    /* === Lazy move generation (sets game_state) === */
    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    /* === Terminal / leaf checks === */
    if(state->game_state == WIN){
        return P_MAX - ply;
    }

    if(state->game_state == DRAW){
        return 0;
    }

    /* === Repetition check (game-specific) === */
    int rep_score;
    if(state->check_repetition(history, rep_score)){
        return rep_score;
    }
    history.push(state->hash());

    if(depth <= 0){
        int score = state->evaluate(
            p.use_kp_eval, p.use_eval_mobility, &history
        );
        history.pop(state->hash());
        return score;
    }

    /* === PVS Negamax loop === */
    int best_score = M_MAX;
    bool first_move = true;

    for(auto& action : state->legal_actions){
        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();
        
        int raw, score;

        if (first_move) {
            // 第一步：使用完整的 Alpha-Beta 視窗進行搜尋
            if(same){
                raw = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, alpha, beta);
            }else{
                raw = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, -beta, -alpha);
            }
            score = same ? raw : -raw;
        } else {
            // 後續步：使用零視窗 (Zero Window) 進行快速驗證
            if(same){
                raw = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, alpha, alpha + 1);
            }else{
                raw = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, -alpha - 1, -alpha);
            }
            score = same ? raw : -raw;

            // Fail-High：零視窗發現這個走法比預期的好 (大於 alpha)，但尚未超出 beta
            // 觸發重新搜尋 (Re-search)，改用完整視窗重新評估
            if (score > alpha && score < beta) {
                if(same){
                    raw = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, score, beta);
                }else{
                    raw = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, -beta, -score);
                }
                score = same ? raw : -raw;
            }
        }

        delete next;

        if(score > best_score){
            best_score = score;
        }

        if(score > alpha){
            alpha = score;
        }

        if(alpha >= beta){
            // 觸發剪枝
            break;
        }
        
        first_move = false;
    }

    history.pop(state->hash());
    return best_score;
}

/*============================================================
 * PVS -- search
 *
 * Iterate legal moves, call eval_ctx with PVS logic,
 * and return SearchResult.
 *============================================================*/
SearchResult PVS::search(
    State *state,
    int depth,
    GameHistory& history,
    SearchContext& ctx
){
    ctx.reset();
    PVSParams p = PVSParams::from_map(ctx.params);
    SearchResult result;
    result.depth = depth;

    if(!state->legal_actions.size()){
        state->get_legal_actions();
    }

    int alpha = M_MAX;
    int beta = P_MAX;
    int best_score = M_MAX - 10;
    int move_index = 0;
    int total_moves = (int)state->legal_actions.size();
    
    bool first_move = true;

    for(auto& action : state->legal_actions){
        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();

        int raw, score;

        if (first_move) {
            if(same){
                raw = eval_ctx(next, depth - 1, history, 1, ctx, p, alpha, beta);
            }else{
                raw = eval_ctx(next, depth - 1, history, 1, ctx, p, -beta, -alpha);
            }
            score = same ? raw : -raw;
        } else {
            // 零視窗搜尋
            if(same){
                raw = eval_ctx(next, depth - 1, history, 1, ctx, p, alpha, alpha + 1);
            }else{
                raw = eval_ctx(next, depth - 1, history, 1, ctx, p, -alpha - 1, -alpha);
            }
            score = same ? raw : -raw;

            // 重新搜尋
            if (score > alpha && score < beta) {
                if(same){
                    raw = eval_ctx(next, depth - 1, history, 1, ctx, p, score, beta);
                }else{
                    raw = eval_ctx(next, depth - 1, history, 1, ctx, p, -beta, -score);
                }
                score = same ? raw : -raw;
            }
        }

        delete next;

        if(score > best_score){
            best_score = score;
            result.best_move = action;

            if(p.report_partial && ctx.on_root_update){
                ctx.on_root_update({result.best_move, best_score, depth, move_index + 1, total_moves});
            }
        }

        if(score > alpha){
            alpha = score;
        }

        if(alpha >= beta){
            break;
        }

        first_move = false;
        move_index++;
    }

    result.score = best_score;
    result.seldepth = ctx.seldepth;
    result.nodes = ctx.nodes;
    if(total_moves > 0){
        result.pv = {result.best_move};
    }

    return result;
}

/*============================================================
 * PVS -- default_params / param_defs
 *============================================================*/
ParamMap PVS::default_params(){
    return {
        {"UseKPEval", "true"},
        {"UseEvalMobility", "true"},
        {"ReportPartial", "true"},
    };
}

std::vector<ParamDef> PVS::param_defs(){
    return {
        {"UseKPEval", ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"ReportPartial", ParamDef::CHECK, "true"},
    };
}