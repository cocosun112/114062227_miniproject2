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
        int score;
        if (p.use_quiescence){
            score = quiescence(state, 0, history, ply, ctx, p, alpha, beta);
        }
        else {
            score = state->evaluate(p.use_kp_eval, p.use_eval_mobility, &history);
        }
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
/*============================================================
 * PVS -- search (Added Iterative Deepening & Time Shield)
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
    
    // 用來儲存「上一層」安全算完的最佳結果
    SearchResult best_result_overall;
    best_result_overall.depth = 0;
    best_result_overall.score = M_MAX - 10;
    best_result_overall.seldepth = 0;
    best_result_overall.nodes = 0;

    if(!state->legal_actions.size()){
        state->get_legal_actions();
    }

    if (state->legal_actions.empty()) {
        return best_result_overall;
    }

    // 防呆機制：預設拿第一個合法步當作保底，避免連深度 1 都沒算完就被 Kill
    best_result_overall.best_move = state->legal_actions[0];
    if (state->legal_actions.size() > 0) {
        best_result_overall.pv = {best_result_overall.best_move};
    }

    // ==========================================================
    // 迭代加深 (Iterative Deepening) 迴圈
    // 從深度 1 開始，一層一層往下算，直到目標 depth 或時間結束
    // ==========================================================
    for (int d = 1; d <= depth; ++d) {
        SearchResult current_depth_result;
        current_depth_result.depth = d;

        int alpha = M_MAX;
        int beta = P_MAX;
        int best_score = M_MAX - 10;
        int move_index = 0;
        int total_moves = (int)state->legal_actions.size();
        
        bool first_move = true;

        // 【優化】根節點步法排序 (Root Move Ordering)
        // 將上一層 (d-1) 找到的 best_move 移到陣列最前面。
        // 這對 PVS 的「零視窗」極度重要，能讓剪枝效率飆升！
        if (d > 1) {
            for (size_t i = 0; i < state->legal_actions.size(); ++i) {
                if (state->legal_actions[i] == best_result_overall.best_move) {
                    std::swap(state->legal_actions[0], state->legal_actions[i]);
                    break;
                }
            }
        }

        for(auto& action : state->legal_actions){
            State* next = state->next_state(action);
            bool same = next->same_player_as_parent();

            int raw, score;

            if (first_move) {
                // 第一步：完整視窗
                if(same){
                    raw = eval_ctx(next, d - 1, history, 1, ctx, p, alpha, beta);
                }else{
                    raw = eval_ctx(next, d - 1, history, 1, ctx, p, -beta, -alpha);
                }
                score = same ? raw : -raw;
            } else {
                // 零視窗搜尋
                if(same){
                    raw = eval_ctx(next, d - 1, history, 1, ctx, p, alpha, alpha + 1);
                }else{
                    raw = eval_ctx(next, d - 1, history, 1, ctx, p, -alpha - 1, -alpha);
                }
                score = same ? raw : -raw;

                // 重新搜尋
                if (score > alpha && score < beta) {
                    if(same){
                        raw = eval_ctx(next, d - 1, history, 1, ctx, p, score, beta);
                    }else{
                        raw = eval_ctx(next, d - 1, history, 1, ctx, p, -beta, -score);
                    }
                    score = same ? raw : -raw;
                }
            }

            delete next;

            // 【保命符 1】如果在算這步的途中時間到了，這個 score 是被污染的垃圾值，絕對不能用！
            if (ctx.stop) {
                break; 
            }

            if(score > best_score){
                best_score = score;
                current_depth_result.best_move = action;

                if(p.report_partial && ctx.on_root_update){
                    ctx.on_root_update({current_depth_result.best_move, best_score, d, move_index + 1, total_moves});
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

        // 【保命符 2】如果這層 d 是因為超時而中斷的，我們絕對不能儲存 current_depth_result
        if (ctx.stop) {
            break; // 跳出 ID 迴圈，安全地回傳上一層 (d-1) 算好的 best_result_overall
        }

        // 這層完整算完了，沒有被時間砍掉，我們安心把結果存起來
        current_depth_result.score = best_score;
        current_depth_result.seldepth = ctx.seldepth;
        current_depth_result.nodes = ctx.nodes;
        if(total_moves > 0){
            current_depth_result.pv = {current_depth_result.best_move};
        }
        
        best_result_overall = current_depth_result;

        // 【提早結束】如果已經找到必勝步 (分數極高，代表即將將死對手)，就不需要再往下算了
        if (best_score >= P_MAX - 100) {
            break;
        }
    }

    return best_result_overall;
}

int PVS::quiescence(
    State *state,
    int qs_depth,
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

    // 1. Stand Pat (維持現狀)：假設我們什麼都不做，直接拿當前盤面分數
    int stand_pat = state->evaluate(p.use_kp_eval, false, &history);
    
    // 如果什麼都不做就已經大於等於 beta，代表對手不會允許這個局面發生，直接剪枝
    if(stand_pat >= beta){
        return beta;
    }
    // 如果當前分數大於 alpha，更新我們的最低保障分數
    if(stand_pat > alpha){
        alpha = stand_pat;
    }

    // 2. 深度限制與終局檢查 (Boss 的 QuiescenceMaxDepth 是 16)
    if(qs_depth >= 16 || state->game_state == WIN || state->game_state == DRAW){
        return stand_pat;
    }

    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    // 3. 展開步法：只展開「吃子步」
    for(auto& action : state->legal_actions){
        
        // 【關鍵】如果不是吃子步，直接跳過不看！
        if(!state->is_capture(action)){
            continue;
        }

        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();

        int score;
        if(same){
            score = quiescence(next, qs_depth + 1, history, ply + 1, ctx, p, alpha, beta);
        } else {
            score = -quiescence(next, qs_depth + 1, history, ply + 1, ctx, p, -beta, -alpha);
        }
        delete next;

        if(score >= beta){
            return beta;
        }
        if(score > alpha){
            alpha = score;
        }
    }
    return alpha;
}

/*============================================================
 * PVS -- default_params / param_defs
 *============================================================*/
ParamMap PVS::default_params(){
    return {
        {"UseKPEval", "true"},
        {"UseEvalMobility", "true"},
        {"ReportPartial", "true"},
        {"UseQuiescence", "true"},
    };
}

std::vector<ParamDef> PVS::param_defs(){
    return {
        {"UseKPEval", ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"ReportPartial", ParamDef::CHECK, "true"},
        {"UseQuiescence", ParamDef::CHECK, "true"},
    };
}